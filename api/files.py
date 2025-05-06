# api/files.py
import os
import io
import time
import uuid
import pandas as pd
from datetime import datetime
from typing import List, Optional

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Depends,
    Path as FastApiPath,  # Use alias to avoid conflict with os.path
    Query,
    Response,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse

from schemas import (
    FileUploadResponse,
    AnalysisStatus,
    AnalysisListItem,
    ReviewRequest,
)
from core.config import UPLOAD_DIR, RESULTS_DIR, get_logger
from core import tasks  # Import tasks module to access jobs and async function
from processing.nlp import process_text
from processing.ml import (
    onnx_session,
    tfidf_vectorizer,
    scaler,
)  # Direct import for sync endpoint
import numpy as np


logger = get_logger(__name__)
router = APIRouter()


# --- File Upload Endpoint ---
@router.post(
    "/uploads/review-file",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["File Analysis"],
)
async def upload_review_file(file: UploadFile = File(...)):
    """
    Uploads a CSV or PDF file containing reviews for asynchronous analysis.

    Supported formats: `text/csv`, `application/pdf`.

    Returns an `upload_id` which can be used to start the analysis process.
    """
    # Validate file type more strictly
    allowed_types = [
        "text/csv",
        "application/pdf",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ]
    if file.content_type not in allowed_types:
        logger.warning(
            f"Upload rejected: Invalid file type '{file.content_type}' for file '{file.filename}'."
        )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Please upload CSV or PDF.",
        )

    # Generate unique ID for the upload
    upload_id = str(uuid.uuid4())
    file_name = file.filename
    # file_extension = ".csv" if file.content_type == "text/csv" else ".pdf"
    # # Sanitize filename briefly (more robust sanitization might be needed)
    # safe_filename = "".join(
    #     c for c in file.filename if c.isalnum() or c in ("_", ".", "-")
    # ).strip()
    # if not safe_filename:
    #     safe_filename = f"uploaded_file{file_extension}"
    # elif not safe_filename.endswith(file_extension):
    #     safe_filename += file_extension

    # Save the file securely
    file_path = os.path.join(UPLOAD_DIR, f"{upload_id}_{file_name}")
    file_size = 0
    try:
        with open(file_path, "wb") as f:
            # Read file in chunks to handle large files
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await file.read(chunk_size):
                f.write(chunk)
                file_size += len(chunk)
        logger.info(
            f"File '{file_name}' uploaded successfully. Size: {file_size} bytes. Path: {file_path}"
        )

    except Exception as e:
        logger.error(f"Error saving uploaded file '{file_name}': {e}")
        # Clean up partial file if saving failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save uploaded file.",
        )

    # Create response
    response = FileUploadResponse(
        upload_id=upload_id,
        filename=file_name,  # Return sanitized filename
        status="uploaded",
        upload_time=datetime.now().isoformat(),
        file_size=file_size,
    )
    return response


# --- Start Analysis Endpoint ---
@router.post(
    "/analysis/process-file/{upload_id}",
    response_model=AnalysisStatus,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["File Analysis"],
)
async def process_uploaded_file(upload_id: str, background_tasks: BackgroundTasks):
    """
    Starts the asynchronous analysis process for a previously uploaded file.

    Requires the `upload_id` obtained from the upload endpoint.
    Returns the initial status of the analysis job.
    """
    # Find the uploaded file matching the upload_id prefix
    upload_file = None
    original_filename = "unknown_file"
    try:
        for f in os.listdir(UPLOAD_DIR):
            if f.startswith(upload_id):
                upload_file = f
                # Extract original filename safely
                parts = f.split("_", 1)
                if len(parts) > 1:
                    original_filename = parts[1]
                else:
                    original_filename = f  # Fallback if no underscore
                break
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Upload directory not found.")
    except Exception as e:
        logger.error(f"Error searching for upload ID {upload_id}: {e}")
        raise HTTPException(status_code=500, detail="Error accessing upload directory.")

    if not upload_file:
        logger.warning(
            f"Analysis request failed: No uploaded file found with ID prefix: {upload_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No uploaded file found with ID: {upload_id}. Please upload the file first.",
        )

    file_path = os.path.join(UPLOAD_DIR, upload_file)
    if not os.path.exists(file_path):
        logger.error(
            f"File path {file_path} not found despite being listed for upload ID {upload_id}."
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Uploaded file inconsistency for ID: {upload_id}. Please try uploading again.",
        )

    # Determine file type based on extension
    file_type = "csv" if upload_file.lower().endswith(".csv") else "pdf"
    file_size = os.path.getsize(file_path)

    # Create an analysis job entry
    analysis_id = str(uuid.uuid4())
    now_iso = datetime.now().isoformat()
    analysis_status_data = {
        "analysis_id": analysis_id,
        "status": "pending",
        "progress": 0.0,
        "error_message": None,
        "result_url": None,
        "created_at": now_iso,
        "updated_at": now_iso,
        "file_size": file_size,
        "file_type": file_type,
        "upload_id": upload_id,
        "file_name": original_filename,  # Store the original filename
        "total_reviews": None,
        "flagged_reviews": None,
        "confidence_score": None,
    }

    # Add job to the in-memory store (using helper function)
    tasks.add_analysis_job(analysis_status_data)

    # Schedule the file processing task to run in the background
    background_tasks.add_task(
        tasks.process_file_async, upload_id, file_path, file_type, analysis_id
    )
    logger.info(
        f"Scheduled background analysis task for upload_id: {upload_id}, analysis_id: {analysis_id}"
    )

    return AnalysisStatus(**analysis_status_data)


# --- Get Analysis Status Endpoint ---
@router.get(
    "/analysis/{analysis_id}", response_model=AnalysisStatus, tags=["File Analysis"]
)
async def get_analysis_status(
    analysis_id: str = FastApiPath(
        ..., description="The unique ID of the analysis job."
    )
):
    """Gets the current status and progress of an analysis job."""
    job = tasks.get_analysis_job(analysis_id)  # Use helper function
    if not job:
        logger.warning(
            f"Analysis status request failed: Job ID not found: {analysis_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis job not found with ID: {analysis_id}",
        )
    return AnalysisStatus(**job)


# --- List Analyses Endpoint ---
@router.get("/analyses", response_model=List[AnalysisListItem], tags=["File Analysis"])
async def get_analyses():
    """Gets a list of all analysis jobs, sorted by creation date (newest first)."""
    all_jobs = tasks.get_all_analysis_jobs()  # Use helper function

    # Adapt the job data to the AnalysisListItem schema
    analyses_list = []
    for job in all_jobs:
        item = AnalysisListItem(
            analysis_id=job.get("analysis_id", "N/A"),
            file_name=job.get("file_name", "Unknown File"),
            created_at=job.get(
                "created_at", datetime.min.isoformat()
            ),  # Use min date if missing
            status=job.get("status", "unknown"),
            analysis_type=job.get("file_type", "unknown").upper(),
            progress=job.get("progress", 0.0),
            updated_at=job.get(
                "updated_at", job.get("created_at", datetime.min.isoformat())
            ),
            error_message=job.get("error_message", None),
        )
        analyses_list.append(item)

    # Sort by creation date, newest first (handle potential missing dates)
    analyses_list.sort(key=lambda x: x.created_at, reverse=True)

    return analyses_list


# --- Download Results Endpoint ---
@router.get("/analysis/{analysis_id}/download", tags=["File Analysis"])
async def download_analysis_results(
    analysis_id: str = FastApiPath(
        ..., description="The unique ID of the completed analysis job."
    ),
    format: str = Query(
        "pdf", description="Download format ('csv' or 'pdf').", regex="^(csv|pdf)$"
    ),
):
    """
    Downloads the results of a completed analysis job as a CSV or PDF file.

    Defaults to PDF format if not specified.
    """
    logger.info(f"Download requested for analysis_id: {analysis_id}, format: {format}")

    job = tasks.get_analysis_job(analysis_id)  # Use helper function
    if not job:
        logger.warning(f"Download request failed: Job ID not found: {analysis_id}")
        raise HTTPException(
            status_code=404, detail=f"Analysis job not found: {analysis_id}"
        )

    if job.get("status") != "completed":
        logger.warning(
            f"Download request failed for job {analysis_id}: Job not completed (status: {job.get('status')})."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Analysis job is not completed yet. Current status: {job.get('status', 'unknown')}",
        )

    # Determine file path based on requested format
    if format.lower() == "pdf":
        file_path = os.path.join(RESULTS_DIR, f"{analysis_id}_report.pdf")
        media_type = "application/pdf"
        # Use original filename from job data for download filename
        base_filename = job.get("file_name", "analysis").rsplit(".", 1)[
            0
        ]  # Get filename without extension
        download_filename = f"{base_filename}_report.pdf"
    else:  # Default to CSV
        file_path = os.path.join(RESULTS_DIR, f"{analysis_id}_results.csv")
        media_type = "text/csv"
        base_filename = job.get("file_name", "analysis").rsplit(".", 1)[0]
        download_filename = f"{base_filename}_results.csv"

    if not os.path.exists(file_path):
        logger.error(
            f"Result file not found for job {analysis_id} at path: {file_path}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result file ({format}) not found for analysis ID: {analysis_id}. Processing might have failed partially.",
        )

    # Use FileResponse for efficient file serving
    logger.info(f"Serving file {file_path} for download as '{download_filename}'.")
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=download_filename,  # Suggests filename to browser
        # headers={"Content-Disposition": f'attachment; filename="{download_filename}"'} # Alternative header setting
    )


# --- Synchronous CSV Prediction Endpoint (Consider Deprecating/Removing if Async preferred) ---
@router.post("/predict_csv_sync", tags=["Prediction", "Deprecated"], deprecated=True)
async def predict_csv_sync(file: UploadFile = File(...)):
    """
    Processes a CSV file synchronously and returns predictions as a downloadable CSV.

    **Deprecated:** Prefer the asynchronous workflow via `/uploads/review-file` and `/analysis/process-file/{upload_id}`.

    CSV must contain a column for review content (e.g., 'review_content', 'text').
    Optional columns for metadata: 'useful_count', 'review_count', 'friend_count'.
    """
    if file.content_type != "text/csv":
        raise HTTPException(
            status_code=415,
            detail="Only CSV files are supported for this synchronous endpoint.",
        )

    try:
        start_time = time.time()
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), on_bad_lines="skip")
        df = df.fillna("")  # Handle missing values

        logger.info(f"Processing sync CSV '{file.filename}' with {len(df)} rows.")

        # --- Column Mapping (same as async task) ---
        content_col = next(
            (
                c
                for c in [
                    "reviewContent",
                    "review_content",
                    "content",
                    "text",
                    "Review Text",
                ]
                if c in df.columns
            ),
            None,
        )
        useful_col = next(
            (
                c
                for c in ["usefulCount", "useful_count", "useful", "Helpful Votes"]
                if c in df.columns
            ),
            None,
        )
        review_count_col = next(
            (
                c
                for c in ["reviewCount", "review_count", "reviews", "Total Reviews"]
                if c in df.columns
            ),
            None,
        )
        friend_col = next(
            (
                c
                for c in ["friendCount", "friend_count", "friends", "Friend Count"]
                if c in df.columns
            ),
            None,
        )

        if not content_col:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain a column for review content (e.g., 'review_content', 'text')",
            )

        # --- Feature Preparation ---
        df["processed_review"] = df[content_col].apply(process_text)
        text_features = tfidf_vectorizer.transform(df["processed_review"])

        numerical_data = []
        for index, row in df.iterrows():
            useful = (
                int(row[useful_col])
                if useful_col
                and pd.notna(row[useful_col])
                and str(row[useful_col]).isdigit()
                else 0
            )
            review_count = (
                int(row[review_count_col])
                if review_count_col
                and pd.notna(row[review_count_col])
                and str(row[review_count_col]).isdigit()
                else 0
            )
            friend_count = (
                int(row[friend_col])
                if friend_col
                and pd.notna(row[friend_col])
                and str(row[friend_col]).isdigit()
                else 0
            )
            numerical_data.append([useful, review_count, friend_count])

        numerical_features_scaled = scaler.transform(numerical_data)

        # Combine features
        combined_features = np.hstack(
            (text_features.toarray(), numerical_features_scaled)
        ).astype(np.float32)

        # --- Run Model ---
        ort_inputs = {onnx_session.get_inputs()[0].name: combined_features}
        predictions_output = onnx_session.run(None, ort_inputs)[
            0
        ]  # Adjust indexing if needed

        # Assuming output is [N, 1] where N is batch size
        raw_predictions = predictions_output.flatten()  # Flatten to 1D array
        df["flagged"] = raw_predictions > 0.5
        df["confidence"] = raw_predictions  # Use raw prediction score as confidence

        # --- Prepare Response CSV ---
        output = io.StringIO()
        # Select columns to return - keep original content + results
        result_cols = [content_col]
        if useful_col:
            result_cols.append(useful_col)
        if review_count_col:
            result_cols.append(review_count_col)
        if friend_col:
            result_cols.append(friend_col)
        result_cols.extend(["flagged", "confidence"])

        df[result_cols].to_csv(output, index=False, encoding="utf-8")
        output.seek(0)

        processing_time = time.time() - start_time
        logger.info(
            f"Finished processing sync CSV '{file.filename}' in {processing_time:.2f} seconds."
        )

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=predictions_{file.filename}"
            },
        )
    except ValueError as ve:  # Catch specific errors like missing columns
        logger.error(f"Value error processing sync CSV '{file.filename}': {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Error processing sync CSV '{file.filename}': {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error processing CSV: {str(e)}"
        )
