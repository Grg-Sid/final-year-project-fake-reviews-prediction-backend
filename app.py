from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import joblib
import onnxruntime as ort
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import io

# Download necessary NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
stopwords_set = set(stopwords.words("english"))

# Load model and transformers
onnx_session = ort.InferenceSession("models/review_flagging_model.onnx")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
scaler = joblib.load("models/scaler.pkl")

app = FastAPI()


class ReviewRequest(BaseModel):
    review_content: str
    useful_count: int
    review_count: int
    friend_count: int


def process_text(text: str) -> str:
    """Cleans and tokenizes text for vectorization."""
    tokens = word_tokenize(text.lower())
    return " ".join(
        [word for word in tokens if word.isalnum() and word not in stopwords_set]
    )


@app.post("/predict")
def predict(review: ReviewRequest):
    """Predicts if a single review should be flagged."""
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


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """Processes a CSV file and returns predictions for each row as a downloadable CSV."""
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Ensure required columns exist
        required_columns = {
            "reviewContent",
            "usefulCount",
            "reviewCount",
            "friendCount",
        }
        if not required_columns.issubset(df.columns):
            raise HTTPException(
                status_code=400, detail=f"CSV must contain columns: {required_columns}"
            )

        # Process text data
        df["processed_review"] = df["reviewContent"].apply(process_text)

        # Transform text and numerical features
        text_features = tfidf_vectorizer.transform(df["processed_review"])
        numerical_features = scaler.transform(
            df[["usefulCount", "reviewCount", "friendCount"]]
        )

        # Combine features
        combined_features = np.hstack(
            (text_features.toarray(), numerical_features)
        ).astype(np.float32)
        ort_inputs = {onnx_session.get_inputs()[0].name: combined_features}

        # Run model
        predictions = onnx_session.run(None, ort_inputs)[0]

        # Add predictions to dataframe
        df["flagged"] = predictions.astype(bool)

        # Convert DataFrame to CSV
        output = io.StringIO()
        df[["reviewContent", "flagged"]].to_csv(output, index=False)
        output.seek(0)

        # Return CSV file as a response
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
