# core/tasks.py
import uuid
import os
import asyncio
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

from schemas import ReviewRequest
from core.config import RESULTS_DIR, get_logger
from processing.ml import predict_single_review  # Import core prediction logic
from processing.file_handling import extract_text_from_pdf  # Import PDF logic
from processing.reporting import generate_pdf_report  # Import report logic
from core.db import (
    insert_analysis_job,
    get_analysis_job_by_id,
    update_analysis_job,
    delete_analysis_job,
    get_all_analysis_jobs_from_db,
)

logger = get_logger(__name__)


# --- Background Task Function ---
async def process_file_async(
    upload_id: str, file_path: str, file_type: str, analysis_id: str
):
    """Process uploaded file asynchronously and update job status."""
    job = get_analysis_job_by_id(analysis_id)
    if not job:
        logger.error(
            f"Analysis job {analysis_id} not found in database during processing."
        )
        return  # Should not happen if called correctly

    try:
        logger.info(
            f"Starting background processing for analysis_id: {analysis_id}, file: {job.get('file_name', 'N/A')}"
        )
        # Update job status to processing
        job["status"] = "processing"
        job["updated_at"] = datetime.now().isoformat()
        update_analysis_job(analysis_id, job)  # Update in DB

        results = []
        total_items = 0
        processed_items = 0

        # --- Process based on file type ---
        if file_type == "csv":
            # Read CSV safely
            try:
                df = pd.read_csv(
                    file_path, on_bad_lines="skip"
                )  # Skip problematic lines
                df = df.fillna("")  # Replace NaN with empty strings for text processing
                total_items = len(df)
                logger.info(f"Read {total_items} rows from CSV: {job.get('file_name')}")
            except Exception as read_e:
                job["status"] = "failed"
                job["error_message"] = f"Failed to read CSV file: {read_e}"
                job["updated_at"] = datetime.now().isoformat()
                update_analysis_job(analysis_id, job)  # Update in DB
                logger.error(f"Failed to read CSV for job {analysis_id}: {read_e}")
                return  # Stop processing

            # Flexible column mapping with fallbacks
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
                raise ValueError(
                    "Could not find a suitable review content column in CSV (e.g., 'review_content', 'text')."
                )

            # Process each row
            for index, row in df.iterrows():
                try:
                    # Safely convert numerical columns, defaulting to 0
                    useful_count = (
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

                    review_request = ReviewRequest(
                        review_content=str(row[content_col]),
                        useful_count=useful_count,
                        review_count=review_count,
                        friend_count=friend_count,
                    )

                    result = predict_single_review(review_request)
                    result["review_content"] = (
                        review_request.review_content
                    )  # Add original content for reporting
                    results.append(result)

                    processed_items += 1
                    # Update progress periodically
                    if processed_items % 20 == 0 or processed_items == total_items:
                        progress = (
                            (processed_items / total_items) * 100
                            if total_items > 0
                            else 100
                        )
                        job["progress"] = round(progress, 2)
                        job["updated_at"] = datetime.now().isoformat()
                        update_analysis_job(analysis_id, job)  # Update in DB
                        # logger.debug(f"Job {analysis_id} progress: {job['progress']:.1f}%")
                        await asyncio.sleep(0.001)  # Yield control briefly

                except Exception as row_e:
                    logger.warning(
                        f"Skipping row {index} in {job.get('file_name')} due to error: {row_e}"
                    )
                    # Optionally add error info to results or skip
                    results.append(
                        {
                            "review_content": str(
                                row.get(content_col, "Error reading content")
                            ),
                            "flagged": False,
                            "confidence": 0.0,
                            "error": f"Processing error: {row_e}",
                        }
                    )
                    processed_items += 1  # Still count as processed for progress
                    continue  # Move to next row

        elif file_type == "pdf":
            # Extract text from PDF
            reviews_text = extract_text_from_pdf(file_path)
            total_items = len(reviews_text)
            logger.info(
                f"Extracted {total_items} potential reviews from PDF: {job.get('file_name')}"
            )

            # Process each extracted review
            for i, review_text in enumerate(reviews_text):
                try:
                    review_request = ReviewRequest(
                        review_content=review_text,
                        # No metadata available from PDF by default
                        useful_count=0,
                        review_count=0,
                        friend_count=0,
                    )

                    result = predict_single_review(review_request)
                    result["review_content"] = (
                        review_request.review_content
                    )  # Add content for reporting
                    results.append(result)

                    processed_items += 1
                    if processed_items % 10 == 0 or processed_items == total_items:
                        progress = (
                            (processed_items / total_items) * 100
                            if total_items > 0
                            else 100
                        )
                        job["progress"] = round(progress, 2)
                        job["updated_at"] = datetime.now().isoformat()
                        update_analysis_job(analysis_id, job)  # Update in DB
                        # logger.debug(f"Job {analysis_id} progress: {job['progress']:.1f}%")
                        await asyncio.sleep(
                            0.005
                        )  # Yield control slightly longer for PDF processing
                except Exception as pdf_row_e:
                    logger.warning(
                        f"Skipping extracted text item {i} from {job.get('file_name')} due to error: {pdf_row_e}"
                    )
                    results.append(
                        {
                            "review_content": (
                                review_text[:100] + "..."
                                if review_text
                                else "Error reading content"
                            ),
                            "flagged": False,
                            "confidence": 0.0,
                            "error": f"Processing error: {pdf_row_e}",
                        }
                    )
                    processed_items += 1
                    continue

        # --- Generate results files ---
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Save as CSV
        result_csv_path = os.path.join(RESULTS_DIR, f"{analysis_id}_results.csv")
        try:
            pd.DataFrame(results).to_csv(result_csv_path, index=False, encoding="utf-8")
            logger.info(f"Generated results CSV: {result_csv_path}")
        except Exception as csv_e:
            logger.error(
                f"Failed to generate results CSV for job {analysis_id}: {csv_e}"
            )
            # Continue to try PDF report

        # Generate PDF report
        result_pdf_path = os.path.join(RESULTS_DIR, f"{analysis_id}_report.pdf")
        pdf_success = generate_pdf_report(results, result_pdf_path)
        if not pdf_success:
            logger.warning(
                f"Failed to generate PDF report for job {analysis_id}. CSV might still be available."
            )
            # Don't fail the whole job, just log the warning

        # --- Update job status to completed ---
        job["status"] = "completed"
        job["progress"] = 100.0
        # Result URL should point to the API endpoint, not the file path directly
        job["result_url"] = (
            f"/api/files/analysis/{analysis_id}/download"  # Relative API path
        )
        job["updated_at"] = datetime.now().isoformat()
        job["total_reviews"] = total_items
        job["flagged_reviews"] = sum(
            1 for result in results if result.get("flagged", False)
        )
        update_analysis_job(analysis_id, job)  # Final update in DB
        logger.info(f"Successfully completed processing for analysis_id: {analysis_id}")

    except ValueError as ve:  # Specific expected errors like missing columns
        logger.error(
            f"Configuration error processing file for job {analysis_id}: {str(ve)}"
        )
        job["status"] = "failed"
        job["error_message"] = str(ve)
        job["updated_at"] = datetime.now().isoformat()
        update_analysis_job(analysis_id, job)  # Update in DB
    except Exception as e:
        logger.exception(
            f"Unexpected error processing file for analysis_id {analysis_id}: {str(e)}"
        )  # Log stack trace
        job["status"] = "failed"
        job["error_message"] = f"An unexpected error occurred: {str(e)}"
        job["updated_at"] = datetime.now().isoformat()
        update_analysis_job(analysis_id, job)  # Update in DB
    finally:
        # Optional: Clean up the uploaded file after processing?
        # Consider adding logic here or in a separate cleanup task
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed uploaded file: {file_path}")
        except Exception as remove_e:
            logger.warning(f"Could not remove upload file {file_path}: {remove_e}")
        # pass  # Keep uploaded file for now, cleanup handled separately


# --- Functions to manage jobs (can be expanded) ---
def add_analysis_job(job_data: Dict[str, Any]) -> str:
    analysis_id = job_data["analysis_id"]
    insert_analysis_job(job_data)  # Store in DB
    logger.info(f"Added new analysis job: {analysis_id}")
    return analysis_id


def get_analysis_job(analysis_id: str) -> Optional[Dict[str, Any]]:
    return get_analysis_job_by_id(analysis_id) 


def get_all_analysis_jobs() -> List[Dict[str, Any]]:
    return get_all_analysis_jobs_from_db() 
