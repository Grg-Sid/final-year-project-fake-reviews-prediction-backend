from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import onnxruntime as ort
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
stopwords = set(stopwords.words("english"))

# Load model
onnx_session = ort.InferenceSession("models/review_flagging_model.onnx")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
scaler = joblib.load("models/scaler.pkl")

app = FastAPI()


class ReviewRequest(BaseModel):
    review_content: str
    useful_count: int
    review_count: int
    friend_count: int


def process_text(text):
    tokens = word_tokenize(text.lower())
    return " ".join(
        [word for word in tokens if word.isalnum() and word not in stopwords]
    )


@app.post("/predict")
def predict(review: ReviewRequest):
    processed_review = process_text(review.review_content)
    text_features = tfidf_vectorizer.transform([processed_review])
    numerical_features = scaler.transform(
        [[review.useful_count, review.review_count, review.friend_count]]
    )

    combined_features = np.hstack((text_features.toarray(), numerical_features)).astype(
        np.float32
    )

    ort_inputs = {onnx_session.get_inputs()[0].name: combined_features}
    prediction = onnx_session.run(None, ort_inputs)[0][0]

    return {"flagged": bool(prediction)}
