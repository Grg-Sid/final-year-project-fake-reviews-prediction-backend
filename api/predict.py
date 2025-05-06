# api/predict.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from schemas import (
    ReviewRequest,
    AnalysisResponse,
    BatchAnalysisRequest,
)  # Import necessary schemas
from processing.ml import predict_single_review  # Import prediction logic
from core.config import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/predict", response_model=AnalysisResponse, tags=["Prediction"])
def predict(review: ReviewRequest):
    """
    Analyzes a single review and predicts if it should be flagged.

    - **review_content**: The text of the review.
    - **useful_count**: Number of users who found the review useful (default 0).
    - **review_count**: Total number of reviews written by the user (default 0).
    - **friend_count**: Number of friends the user has (default 0).
    """
    try:
        logger.info(
            f"Received single prediction request: {review.review_content[:50]}..."
        )
        result = predict_single_review(review)
        if "error" in result:  # Check if prediction function indicated an error
            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {result['error']}"
            )
        return AnalysisResponse(**result)
    except Exception as e:
        logger.exception(
            f"Error in single prediction endpoint: {str(e)}"
        )  # Log stack trace
        raise HTTPException(
            status_code=500, detail=f"Internal server error during prediction: {str(e)}"
        )


@router.post(
    "/batch_predict", response_model=List[AnalysisResponse], tags=["Prediction"]
)
def batch_predict(request: BatchAnalysisRequest):
    """
    Analyzes a batch of reviews in a single request.

    Provides a list of reviews in the same format as the single `/predict` endpoint.
    """
    results = []
    logger.info(
        f"Received batch prediction request with {len(request.reviews)} reviews."
    )
    from processing.ml import (
        predict_batch_reviews_optimized,
    )

    return predict_batch_reviews_optimized([r.dict() for r in request.reviews])

    # for i, review in enumerate(request.reviews):
    #     try:
    #         result = predict_single_review(review)
    #         if "error" in result:
    #             logger.warning(
    #                 f"Error predicting review {i} in batch: {result['error']}"
    #             )
    #             # Decide how to handle errors in batch: skip, return error marker, etc.
    #             # Skipping for now, could append an error marker if needed
    #             continue
    #         results.append(AnalysisResponse(**result))
    #     except Exception as e:
    #         # Log error for this specific review but continue with batch
    #         logger.error(
    #             f"Failed to process review {i} in batch: {review.review_content[:50]}... Error: {e}"
    #         )
    #         continue  # Skip review on unexpected error

    # logger.info(
    #     f"Completed batch prediction request. Returning {len(results)} results."
    # )
    # return results
