# processing/ml.py
import numpy as np
import joblib
import onnxruntime as ort
import time
from typing import Dict, Any, List
from schemas import ReviewRequest  # Use schema for type hinting
from core.config import (
    ONNX_MODEL_PATH,
    TFIDF_VECTORIZER_PATH,
    SCALER_PATH,
    get_logger,
)
from processing.nlp import process_text  # Import needed function

logger = get_logger(__name__)

# --- Load model and transformers ---
try:
    onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("ML Model and transformers loaded successfully")
except FileNotFoundError as e:
    logger.error(
        f"Model file not found: {e}. Ensure models are in the correct directory."
    )
    raise
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise


# --- Prediction Function ---
def predict_single_review(review: ReviewRequest) -> Dict[str, Any]:
    """Predicts if a single review should be flagged."""
    start_time = time.time()

    processed_review = process_text(review.review_content)
    text_features = tfidf_vectorizer.transform([processed_review])
    numerical_features = scaler.transform(
        [[review.useful_count, review.review_count, review.friend_count]]
    )
    combined_features = np.hstack((text_features.toarray(), numerical_features)).astype(
        np.float32
    )
    ort_inputs = {onnx_session.get_inputs()[0].name: combined_features}

    # Get both prediction and probability
    prediction = onnx_session.run(None, ort_inputs)[0][0]
    confidence = float(
        abs(prediction)
    )  # Using absolute value as a simple confidence measure

    process_time = time.time() - start_time

    return {
        "flagged": bool(prediction > 0.5),  # Assuming threshold of 0.5
        "confidence": confidence,
        "process_time": process_time,
    }


def predict_batch_reviews_optimized(reviews_data: List[Dict]) -> List[Dict]:
    """Optimized function to predict multiple reviews using batch processing."""
    start_time = time.time()

    try:
        # 1. Preprocess text data in batch
        processed_texts = [
            process_text(review.get("review_content", "")) for review in reviews_data
        ]
        text_features = tfidf_vectorizer.transform(processed_texts)

        # 2. Prepare numerical data in batch
        numerical_data = [
            [
                review.get("useful_count", 0),
                review.get("review_count", 0),
                review.get("friend_count", 0),
            ]
            for review in reviews_data
        ]
        numerical_features_scaled = scaler.transform(numerical_data)

        # 3. Combine features
        combined_features = np.hstack(
            (text_features.toarray(), numerical_features_scaled)
        ).astype(np.float32)

        # 4. Prepare ONNX input
        ort_inputs = {onnx_session.get_inputs()[0].name: combined_features}

        # 5. Run ONNX inference
        model_outputs = onnx_session.run(None, ort_inputs)[
            0
        ]  # Assuming first output contains predictions

        # 6. Process results
        results = []
        total_process_time = time.time() - start_time
        per_item_process_time = (
            total_process_time / len(reviews_data) if reviews_data else 0
        )

        for i, output in enumerate(model_outputs):
            raw_prediction = output[
                0
            ]  # Adjust indexing based on actual model output shape
            flagged = bool(raw_prediction > 0.5)
            confidence = float(raw_prediction)
            results.append(
                {
                    "flagged": flagged,
                    "confidence": confidence,
                    "process_time": per_item_process_time,  # Approximate time per item
                    # Optionally include original review identifier if passed in reviews_data
                    "original_index": i,  # Example
                }
            )

        return results

    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        # Return empty list or error indicators for all items
        return [{"error": str(e)} for _ in reviews_data]
