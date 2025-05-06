# core/config.py
import os
import logging

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# --- Model Paths ---
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "review_flagging_model.onnx")
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# --- Logging Configuration ---
LOGGING_LEVEL = logging.INFO
# Basic config is set in main.py, but level is defined here

# --- Cleanup Configuration ---
CLEANUP_FILE_AGE_SECONDS = 86400  # 24 hours

# --- Ensure Directories Exist ---
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- Helper to get logger ---
def get_logger(name: str):
    return logging.getLogger(name)
