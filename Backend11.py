import atexit
import io
import json
import logging
import os
import zipfile
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import joblib
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from gotrue.errors import AuthRetryableError
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Session, relationship
from supabase import Client, create_client

from database import Base, SessionLocal, engine, get_db

load_dotenv('.env')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('theriax-backend')

app = FastAPI(title='Theriax Backend')


def parse_allowed_origins(raw_origins: Optional[str]) -> List[str]:
    if not raw_origins:
        return ['http://localhost:5173', 'http://127.0.0.1:5173']
    origins = [origin.strip() for origin in raw_origins.split(',') if origin.strip()]
    return origins or ['http://localhost:5173', 'http://127.0.0.1:5173']


ALLOWED_ORIGINS = parse_allowed_origins(os.getenv('ALLOWED_ORIGINS'))
ALLOW_ALL_ORIGINS = '*' in ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'] if ALLOW_ALL_ORIGINS else ALLOWED_ORIGINS,
    allow_credentials=not ALLOW_ALL_ORIGINS,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError('SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env')

EVOLVE_URL = os.getenv('EVOLVE_URL', 'http://localhost:9000/predict')
GOOGLE_REDIRECT_URL = os.getenv('GOOGLE_REDIRECT_URL', 'http://localhost:5173/auth/callback')
ENABLE_RETRAIN_SCHEDULER = os.getenv('ENABLE_RETRAIN_SCHEDULER', 'false').lower() in {'1', 'true', 'yes'}
MIN_TRAINING_ROWS = int(os.getenv('MIN_TRAINING_ROWS', '25'))

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
bearer_scheme = HTTPBearer(auto_error=False)


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String, nullable=True)


class Condition(Base):
    __tablename__ = 'conditions'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    medicines = relationship('Medicine', back_populates='condition')


class Medicine(Base):
    __tablename__ = 'medicines'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    base_dosage = Column(String)
    is_generic = Column(Boolean)
    cost = Column(Float)
    min_severity = Column(String)
    condition_id = Column(Integer, ForeignKey('conditions.id'))
    condition = relationship('Condition', back_populates='medicines')


class AIPredictionLog(Base):
    __tablename__ = 'ai_prediction_logs'
    id = Column(Integer, primary_key=True, index=True)
    symptoms = Column(String)
    age = Column(Integer)
    weight = Column(Float)
    severity = Column(String)
    condition = Column(String)
    predicted_medicine = Column(String)
    predicted_dosage = Column(String, nullable=True)
    predicted_cost = Column(Float, nullable=True)
    is_generic = Column(Boolean, nullable=True)
    confidence = Column(Float, nullable=True)
    top_predictions = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class RetrainTracker(Base):
    __tablename__ = 'retrain_tracker'
    id = Column(Integer, primary_key=True, index=True)
    last_retrained_at = Column(DateTime, default=datetime.utcnow)


class AuthDetails(BaseModel):
    email: str = Field(min_length=5)
    password: str = Field(min_length=6)


class Case(BaseModel):
    symptoms: str = Field(min_length=1)
    age: int = Field(ge=0, le=120)
    weight: float = Field(gt=0, le=500)
    severity: str = Field(min_length=1)
    condition: str = Field(min_length=1)


def parse_symptom_list(symptoms: str) -> List[str]:
    return [symptom.strip() for symptom in (symptoms or '').split(',') if symptom.strip()]


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
    if credentials is None or credentials.scheme.lower() != 'bearer':
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Missing bearer token')

    token = credentials.credentials
    try:
        user_response = supabase.auth.get_user(jwt=token)
        if not user_response or not getattr(user_response, 'user', None):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token')
        return user_response.user
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning('Token verification failed: %s', exc)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid or expired token')


def ensure_local_user(email: str, full_name: Optional[str], db: Session) -> User:
    db_user = db.query(User).filter(User.email == email).first()
    if not db_user:
        db_user = User(email=email, full_name=full_name)
        db.add(db_user)
    elif full_name and db_user.full_name != full_name:
        db_user.full_name = full_name
    db.commit()
    db.refresh(db_user)
    return db_user


def zip_pickle_content(obj, inner_filename: str) -> bytes:
    pickle_bytes = io.BytesIO()
    joblib.dump(obj, pickle_bytes)
    pickle_bytes.seek(0)

    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(inner_filename, pickle_bytes.read())

    return zip_bytes.getvalue()


def upload_model_artifact(path: str, content: bytes, bucket: str = 'models') -> None:
    storage = supabase.storage.from_(bucket)
    try:
        storage.remove([path])
    except Exception:
        pass

    storage.upload(
        path=path,
        file=content,
        file_options={
            'content-type': 'application/zip',
            'upsert': 'true',
        },
    )


def train_and_upload_from_logs(logs: List[AIPredictionLog]) -> Dict[str, int]:
    dataset = []
    for log in logs:
        dataset.append(
            {
                'symptoms': log.symptoms or '',
                'age': log.age,
                'weight': log.weight,
                'severity': (log.severity or '').strip().lower(),
                'condition': (log.condition or '').strip(),
                'medicine': (log.predicted_medicine or '').strip(),
            }
        )

    data = pd.DataFrame(dataset)
    data = data.dropna(subset=['age', 'weight'])
    data = data[data['medicine'].astype(str).str.len() > 0]
    data = data[data['severity'].astype(str).str.len() > 0]
    data = data[data['condition'].astype(str).str.len() > 0]

    if len(data) < MIN_TRAINING_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f'Need at least {MIN_TRAINING_ROWS} usable rows for retraining; found {len(data)}.',
        )

    if data['medicine'].nunique() < 2:
        raise HTTPException(status_code=400, detail='Need at least two medicine classes for retraining.')

    data['symptom_list'] = data['symptoms'].apply(parse_symptom_list)
    data = data[data['symptom_list'].map(bool)].reset_index(drop=True)
    if data.empty:
        raise HTTPException(status_code=400, detail='No valid symptom lists found for retraining.')

    symptom_binarizer = MultiLabelBinarizer()
    symptom_matrix = symptom_binarizer.fit_transform(data['symptom_list'])
    symptom_columns = list(symptom_binarizer.classes_)
    symptom_encoded = pd.DataFrame(symptom_matrix, columns=symptom_columns)

    prepared = pd.concat([data.reset_index(drop=True), symptom_encoded], axis=1)

    encoders: Dict[str, LabelEncoder] = {}
    for column in ['severity', 'condition', 'medicine']:
        encoder = LabelEncoder()
        prepared[column] = encoder.fit_transform(prepared[column].astype(str))
        encoders[column] = encoder

    features = prepared[symptom_columns + ['age', 'weight', 'severity', 'condition']]
    target = prepared['medicine']

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=8,
        class_weight='balanced',
        random_state=42,
    )
    model.fit(features, target)

    model_zip = zip_pickle_content(model, 'latest_model.pkl')
    encoders_zip = zip_pickle_content(encoders, 'latest_encoders.pkl')
    binarizer_zip = zip_pickle_content(symptom_binarizer, 'latest_binarizer.pkl')

    upload_model_artifact('latest_model.zip', model_zip)
    upload_model_artifact('latest_encoders.zip', encoders_zip)
    upload_model_artifact('latest_binarizer.zip', binarizer_zip)

    return {
        'trained_rows': int(len(features)),
        'classes': int(target.nunique()),
        'symptom_features': int(len(symptom_columns)),
    }


def retrain_model(db: Session) -> Dict[str, int]:
    logs = db.query(AIPredictionLog).all()
    if not logs:
        raise HTTPException(status_code=404, detail='No prediction logs found for retraining.')

    summary = train_and_upload_from_logs(logs)

    tracker = db.query(RetrainTracker).order_by(RetrainTracker.id.desc()).first()
    now = datetime.utcnow()
    if tracker:
        tracker.last_retrained_at = now
    else:
        tracker = RetrainTracker(last_retrained_at=now)
        db.add(tracker)
    db.commit()

    return summary


@app.post('/signup', status_code=status.HTTP_201_CREATED)
def signup(auth: AuthDetails, db: Session = Depends(get_db)):
    email = auth.email.strip().lower()
    try:
        result = supabase.auth.sign_up({'email': email, 'password': auth.password})
        user = getattr(result, 'user', None)
        if not user:
            raise HTTPException(status_code=400, detail='Signup failed. Please verify the email and password.')

        metadata = getattr(user, 'user_metadata', {}) or {}
        full_name = metadata.get('full_name')
        ensure_local_user(email=email, full_name=full_name, db=db)

        return {'message': 'Signup successful. Check email to confirm your account.'}
    except HTTPException:
        raise
    except Exception as exc:
        error_text = str(exc)
        if 'already registered' in error_text.lower():
            raise HTTPException(status_code=409, detail='User already exists.')
        logger.error('Signup failed: %s', exc)
        raise HTTPException(status_code=500, detail='Unexpected signup error')


@app.post('/login')
def login(auth: AuthDetails, db: Session = Depends(get_db)):
    email = auth.email.strip().lower()
    try:
        result = supabase.auth.sign_in_with_password({'email': email, 'password': auth.password})

        user = getattr(result, 'user', None)
        session = getattr(result, 'session', None)

        if not user or not session:
            raise HTTPException(status_code=401, detail='Invalid email or password.')

        if not session.access_token:
            raise HTTPException(status_code=403, detail='Email not confirmed or no active session.')

        metadata = getattr(user, 'user_metadata', {}) or {}
        db_user = ensure_local_user(email=user.email, full_name=metadata.get('full_name'), db=db)

        return {
            'access_token': session.access_token,
            'refresh_token': session.refresh_token,
            'user': {'email': user.email, 'full_name': db_user.full_name},
        }

    except HTTPException:
        raise
    except AuthRetryableError:
        raise HTTPException(status_code=503, detail='Auth provider timed out. Please retry.')
    except Exception as exc:
        message = str(exc)
        if 'email not confirmed' in message.lower():
            raise HTTPException(status_code=403, detail='Email not confirmed.')
        raise HTTPException(status_code=401, detail='Login failed. Check your credentials.')


@app.get('/auth/google-login-url')
def get_google_login_url():
    url = f'{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={GOOGLE_REDIRECT_URL}'
    return {'url': url}


@app.get('/auth/callback')
def google_auth_callback(request: Request, db: Session = Depends(get_db)):
    token = request.query_params.get('access_token')
    if not token:
        raise HTTPException(status_code=400, detail='Missing access token from callback query.')

    try:
        user_response = supabase.auth.get_user(jwt=token)
        user = getattr(user_response, 'user', None)
        if not user or not user.email:
            raise HTTPException(status_code=401, detail='Invalid token payload.')

        metadata = getattr(user, 'user_metadata', {}) or {}
        ensure_local_user(email=user.email, full_name=metadata.get('full_name'), db=db)

        return {'message': 'Google login successful', 'email': user.email, 'name': metadata.get('full_name', '')}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail='Invalid token')


@app.get('/dashboard')
def dashboard(
    request: Request,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_user),
):
    logs = db.query(AIPredictionLog).order_by(AIPredictionLog.timestamp.desc()).limit(50).all()
    return templates.TemplateResponse('dashboard.html', {'request': request, 'logs': logs})


@app.get('/top-medicines')
def top_medicines(
    db: Session = Depends(get_db),
    _: object = Depends(get_current_user),
):
    top = (
        db.query(AIPredictionLog.predicted_medicine, func.count().label('count'))
        .group_by(AIPredictionLog.predicted_medicine)
        .order_by(func.count().desc())
        .limit(10)
        .all()
    )
    return [{'medicine': med, 'count': count} for med, count in top]


@app.get('/dashboard/top-medicines')
def top_meds_view(
    request: Request,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_user),
):
    top = (
        db.query(AIPredictionLog.predicted_medicine, func.count().label('count'))
        .group_by(AIPredictionLog.predicted_medicine)
        .order_by(func.count().desc())
        .limit(10)
        .all()
    )
    return templates.TemplateResponse('top_medicines.html', {'request': request, 'top_meds': top})


@app.post('/ai-recommend')
def ai_recommend(
    case: Case,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_user),
):
    try:
        response = httpx.post(EVOLVE_URL, json=case.model_dump(), timeout=httpx.Timeout(60.0))
        response.raise_for_status()
        ai_prediction = response.json()
    except httpx.HTTPError as exc:
        logger.error('AI service call failed: %s', exc)
        raise HTTPException(status_code=502, detail='AI service unavailable')
    except Exception as exc:
        logger.error('AI service error: %s', exc)
        raise HTTPException(status_code=500, detail='Unexpected AI model error')

    recommendations = ai_prediction.get('recommendations') or []
    notes = ai_prediction.get('notes', 'AI recommendation generated.')

    top_predictions = []
    for pred in ai_prediction.get('top_predictions', []):
        top_predictions.append(
            {
                'name': pred.get('name'),
                'confidence': float(round(pred.get('confidence', 0.0), 4)),
            }
        )

    confidence = float(ai_prediction.get('confidence', 0.0) or 0.0)

    if recommendations:
        best_recommendation = recommendations[0]
    elif top_predictions:
        best_recommendation = {'name': top_predictions[0]['name']}
    else:
        best_recommendation = {'name': 'Unknown'}

    predicted_medicine = best_recommendation.get('name', 'Unknown')

    medicine_record = db.query(Medicine).filter(func.lower(Medicine.name) == predicted_medicine.lower()).first()

    predicted_dosage = best_recommendation.get('dosage')
    predicted_cost = best_recommendation.get('cost')
    is_generic = best_recommendation.get('is_generic')

    if medicine_record:
        predicted_dosage = medicine_record.base_dosage
        predicted_cost = medicine_record.cost
        is_generic = medicine_record.is_generic

    db_log = AIPredictionLog(
        symptoms=case.symptoms,
        age=case.age,
        weight=case.weight,
        severity=case.severity,
        condition=case.condition,
        predicted_medicine=predicted_medicine,
        predicted_dosage=predicted_dosage,
        predicted_cost=predicted_cost,
        is_generic=is_generic,
        confidence=confidence,
        top_predictions=json.dumps(top_predictions),
    )
    db.add(db_log)
    db.commit()

    return {
        'ai_model': predicted_medicine,
        'info': notes,
        'confidence': confidence,
        'top_predictions': top_predictions,
        'recommendations': recommendations,
    }


@app.post('/retrain-model')
def retrain_model_endpoint(
    db: Session = Depends(get_db),
    _: object = Depends(get_current_user),
):
    summary = retrain_model(db)
    return {'message': 'Model retrained and uploaded successfully.', **summary}


@app.get('/medicines-by-condition')
def get_medicines_by_condition(condition: str, severity: Optional[str] = None, db: Session = Depends(get_db)):
    condition_entry = db.query(Condition).filter(Condition.name == condition).first()
    if not condition_entry:
        raise HTTPException(status_code=404, detail='Condition not found.')

    query = db.query(Medicine).filter(Medicine.condition_id == condition_entry.id)

    if severity:
        severity_order = {'low': 1, 'medium': 2, 'high': 3}
        user_level = severity_order.get(severity.lower())
        if user_level:
            allowed_severities = [sev for sev, level in severity_order.items() if level <= user_level]
            query = query.filter(func.lower(Medicine.min_severity).in_(allowed_severities))

    medicines = query.all()
    return [
        {'name': m.name, 'dosage': m.base_dosage, 'cost': m.cost, 'is_generic': m.is_generic}
        for m in medicines
    ]


@app.get('/dashboard-logs')
def dashboard_logs(
    db: Session = Depends(get_db),
    _: object = Depends(get_current_user),
):
    logs = db.query(AIPredictionLog).order_by(AIPredictionLog.timestamp.desc()).all()
    return [
        {
            'id': log.id,
            'condition': log.condition,
            'symptoms': log.symptoms,
            'medicine': log.predicted_medicine,
            'predicted_medicine': log.predicted_medicine,
            'severity': log.severity,
            'confidence': log.confidence,
            'top_predictions': json.loads(log.top_predictions) if log.top_predictions else [],
            'timestamp': log.timestamp.isoformat() if log.timestamp else None,
        }
        for log in logs
    ]


@app.get('/conditions')
def get_conditions(db: Session = Depends(get_db)):
    conditions = db.query(Condition).order_by(Condition.name.asc()).all()
    return [{'id': c.id, 'name': c.name} for c in conditions]


@app.get('/symptoms')
def get_symptoms(db: Session = Depends(get_db)):
    symptom_rows = db.query(AIPredictionLog.symptoms).distinct().all()

    unique_symptoms = set()
    for row in symptom_rows:
        if row[0]:
            for symptom in parse_symptom_list(row[0]):
                unique_symptoms.add(symptom)

    return sorted(unique_symptoms)


def scheduled_retrain() -> None:
    db = SessionLocal()
    try:
        summary = retrain_model(db)
        logger.info('Scheduled retrain complete: %s', summary)
    except Exception as exc:
        logger.error('Scheduled retrain failed: %s', exc)
    finally:
        db.close()


@app.on_event('startup')
def init_database() -> None:
    try:
        Base.metadata.create_all(bind=engine)
        logger.info('Database schema check complete.')
    except Exception as exc:
        logger.error('Database initialization failed: %s', exc)


@app.on_event('startup')
def start_scheduler() -> None:
    if not ENABLE_RETRAIN_SCHEDULER:
        logger.info('Retrain scheduler disabled.')
        return

    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_retrain, 'interval', days=1)
    scheduler.start()
    app.state.scheduler = scheduler
    atexit.register(lambda: scheduler.shutdown())
    logger.info('Retrain scheduler started.')

