import io
import logging
import os
import shutil
import stat
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client

load_dotenv('.env')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('theriax-evolve')

app = FastAPI(title='Theriax Evolve Service')

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError('SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env')

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000').rstrip('/')
BACKEND_SERVICE_TOKEN = os.getenv('BACKEND_SERVICE_TOKEN')
MODEL_UPDATE_INTERVAL = int(os.getenv('MODEL_UPDATE_INTERVAL_SECONDS', '3600'))
MODEL_DIR = Path('model_files')

model = None
encoders = None
binarizer = None
model_last_updated = None


class Case(BaseModel):
    symptoms: str = Field(min_length=1)
    age: int = Field(ge=0, le=120)
    weight: float = Field(gt=0, le=500)
    severity: str = Field(min_length=1)
    condition: str = Field(min_length=1)


def remove_readonly(func, path, _exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def reset_model_directory() -> None:
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR, onerror=remove_readonly)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def extract_zip_to_model_dir(zip_bytes: bytes, expected_file: str) -> Path:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_ref:
        members = zip_ref.namelist()
        if expected_file not in members:
            raise HTTPException(status_code=500, detail=f'Missing {expected_file} in downloaded artifact.')
        zip_ref.extract(expected_file, MODEL_DIR)
    return MODEL_DIR / expected_file


def download_model_files() -> tuple[Path, Path, Path]:
    reset_model_directory()

    logger.info('Downloading model artifacts from Supabase Storage.')

    model_zip = supabase.storage.from_('models').download('latest_model.zip')
    encoders_zip = supabase.storage.from_('models').download('latest_encoders.zip')
    binarizer_zip = supabase.storage.from_('models').download('latest_binarizer.zip')

    model_path = extract_zip_to_model_dir(model_zip, 'latest_model.pkl')
    encoders_path = extract_zip_to_model_dir(encoders_zip, 'latest_encoders.pkl')
    binarizer_path = extract_zip_to_model_dir(binarizer_zip, 'latest_binarizer.pkl')

    return model_path, encoders_path, binarizer_path


def needs_model_refresh() -> bool:
    if model_last_updated is None:
        return True
    return (time.time() - model_last_updated) > MODEL_UPDATE_INTERVAL


def get_model_and_encoders():
    global model, encoders, binarizer, model_last_updated

    if model is None or encoders is None or binarizer is None or needs_model_refresh():
        model_path, encoders_path, binarizer_path = download_model_files()

        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        binarizer = joblib.load(binarizer_path)
        model_last_updated = time.time()

        required_encoders = {'severity', 'condition', 'medicine'}
        missing = [key for key in required_encoders if key not in encoders]
        if missing:
            raise HTTPException(status_code=500, detail=f'Missing encoder keys: {missing}')

    return model, encoders, binarizer


def encode_with_label_encoder(label_encoder, value: str, field_name: str) -> int:
    normalized_input = value.strip().lower()
    lower_map = {str(item).lower(): item for item in label_encoder.classes_}

    if normalized_input not in lower_map:
        raise HTTPException(status_code=400, detail=f'Unknown {field_name}: {value}')

    return int(label_encoder.transform([lower_map[normalized_input]])[0])


def fetch_medicines_for_condition(condition: str, severity: str):
    headers = {}
    if BACKEND_SERVICE_TOKEN:
        headers['Authorization'] = f'Bearer {BACKEND_SERVICE_TOKEN}'

    response = requests.get(
        f'{BACKEND_URL}/medicines-by-condition',
        params={'condition': condition, 'severity': severity},
        headers=headers,
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model_loaded': model is not None,
        'last_model_refresh': model_last_updated,
    }


@app.post('/reload-model')
def reload_model():
    global model, encoders, binarizer, model_last_updated

    model = None
    encoders = None
    binarizer = None
    model_last_updated = None
    get_model_and_encoders()

    return {'status': 'Model reloaded successfully.'}


@app.post('/predict')
def predict(case: Case):
    try:
        current_model, current_encoders, current_binarizer = get_model_and_encoders()

        symptoms = [symptom.strip() for symptom in case.symptoms.split(',') if symptom.strip()]
        if not symptoms:
            raise HTTPException(status_code=400, detail='No symptoms provided.')

        known_symptoms = set(current_binarizer.classes_)
        valid_symptoms = [symptom for symptom in symptoms if symptom in known_symptoms]
        unknown_symptoms = [symptom for symptom in symptoms if symptom not in known_symptoms]

        if not valid_symptoms:
            raise HTTPException(status_code=400, detail='All provided symptoms are unknown to the model.')

        symptom_vector = current_binarizer.transform([valid_symptoms])[0]
        severity_encoded = encode_with_label_encoder(current_encoders['severity'], case.severity, 'severity')
        condition_encoded = encode_with_label_encoder(current_encoders['condition'], case.condition, 'condition')

        model_input = np.concatenate(
            [symptom_vector, np.array([case.age, case.weight, severity_encoded, condition_encoded])]
        ).reshape(1, -1)

        probabilities = current_model.predict_proba(model_input)[0]
        top_indices = np.argsort(probabilities)[-3:][::-1]

        top_predictions = []
        for index in top_indices:
            label = current_model.classes_[index]
            try:
                label_for_decoder = int(label)
            except (TypeError, ValueError):
                label_for_decoder = label

            medicine_name = current_encoders['medicine'].inverse_transform([label_for_decoder])[0]
            top_predictions.append(
                {
                    'name': medicine_name,
                    'confidence': float(round(float(probabilities[index]), 4)),
                }
            )

        predicted_name = top_predictions[0]['name']
        confidence = top_predictions[0]['confidence']

        backend_medicines = fetch_medicines_for_condition(case.condition, case.severity)
        matched = [medicine for medicine in backend_medicines if medicine['name'].lower() == predicted_name.lower()]

        return {
            'recommendations': matched[:1],
            'notes': (
                'Recommended by AI and matched with backend.'
                if matched
                else f"'{predicted_name}' not available for this condition/severity in backend data."
            ),
            'confidence': confidence,
            'top_predictions': top_predictions,
            'all_backend_medicines': backend_medicines,
            'unmatched_predictions': [
                prediction
                for prediction in top_predictions
                if prediction['name'].lower() not in [medicine['name'].lower() for medicine in backend_medicines]
            ],
            'unknown_symptoms': unknown_symptoms,
        }

    except HTTPException:
        raise
    except requests.RequestException as exc:
        logger.error('Backend medicine lookup failed: %s', exc)
        raise HTTPException(status_code=502, detail='Unable to query backend medicine catalog.')
    except Exception as exc:
        logger.error('Prediction error: %s', exc)
        raise HTTPException(status_code=500, detail='Prediction failed due to internal error.')
