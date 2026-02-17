import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from supabase import create_client
from dotenv import load_dotenv
import zipfile
from pathlib import Path

# --- Load Supabase creds ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_to_supabase(local_path, remote_name, bucket="models"):
    # Read file data
    with open(local_path, "rb") as f:
        file_data = f.read()

    # Delete existing file (to simulate upsert behavior)
    try:
        supabase.storage.from_(bucket).remove([remote_name])
        print(f"Removed existing {remote_name} before uploading.")
    except Exception as e:
        print(f"Couldn't remove {remote_name}, maybe it didn't exist. Proceeding...")

    # Upload new version
    try:
        res = supabase.storage.from_(bucket).upload(
            path=remote_name,
            file=file_data,
            file_options={"content-type": "application/octet-stream"},
        )
        print(f"Uploaded to Supabase: {remote_name} => {res}")
        return True
    except Exception as e:
        print(f"Upload failed for {remote_name}. Kept local artifact. Error: {e}")
        return False

def compress_model(model_path):
    """Compress the model file into a .zip archive."""
    zip_filename = model_path + ".zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(model_path, os.path.basename(model_path))
    return zip_filename

def train_model():
    # Load dataset
    dataset_path = Path(__file__).resolve().parents[1] / "dummy_data.csv"
    df = pd.read_csv(dataset_path)

    # --- Feature Engineering ---
    df["symptom_list"] = df["symptoms"].str.split(", ")
    
    # Encode multi-label symptoms
    mlb = MultiLabelBinarizer()
    symptom_encoded = pd.DataFrame(mlb.fit_transform(df["symptom_list"]), columns=mlb.classes_)
    df = pd.concat([df, symptom_encoded], axis=1)
    df.drop(["symptoms", "symptom_list"], axis=1, inplace=True)

    # Encode other categorical columns
    label_encoders = {}
    for col in ["severity", "condition", "medicine"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Features and target
    symptom_cols = list(symptom_encoded.columns)
    X = df[symptom_cols + ["age", "weight", "severity", "condition"]]
    y = df["medicine"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost model training
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=12,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

    # Validation metrics
    y_pred = model.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # Save model and encoders
    model_filename = "latest_model.pkl"
    encoders_filename = "latest_encoders.pkl"
    symptom_binarizer_filename = "latest_binarizer.pkl"

    joblib.dump(model, model_filename)
    joblib.dump(label_encoders, encoders_filename)
    joblib.dump(mlb, symptom_binarizer_filename)
    print("\nModel, encoders, and symptom binarizer saved successfully.")

    # Compress the model files
    compressed_model = compress_model(model_filename)
    compressed_encoders = compress_model(encoders_filename)
    compressed_binarizer = compress_model(symptom_binarizer_filename)

    # Upload compressed files to Supabase
    upload_to_supabase(compressed_model, "latest_model.zip")
    upload_to_supabase(compressed_encoders, "latest_encoders.zip")
    upload_to_supabase(compressed_binarizer, "latest_binarizer.zip")

if __name__ == "__main__":
    train_model()
