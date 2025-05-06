import sqlite3
import os
import json
from typing import Dict, Any, Optional, List
from core.config import BASE_DIR, get_logger

logger = get_logger(__name__)

DATABASE_PATH = os.path.join(BASE_DIR, "reviews.db")


def get_db_connection():
    """Establishes and returns a database connection."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row  # Access columns by name
        logger.info("Successfully connected to the database.")
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        if conn:
            conn.close()
        raise  # Re-raise the exception to prevent the app from running
    return conn


def create_tables():
    """Creates the necessary database tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_content TEXT NOT NULL,
                flagged BOOLEAN,
                confidence REAL,
                upload_id TEXT,
                analysis_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Add the analysis_jobs table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_jobs (
                analysis_id TEXT PRIMARY KEY,
                file_name TEXT,
                status TEXT,
                progress REAL,
                result_url TEXT,
                error_message TEXT,
                updated_at TEXT,
                total_reviews INTEGER,
                flagged_reviews INTEGER,
                upload_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
        logger.info("Database tables created or already exist.")
    except sqlite3.Error as e:
        logger.error(f"Error creating tables: {e}")
    finally:
        cursor.close()
        conn.close()


def insert_review(
    review_content: str,
    flagged: bool,
    confidence: float,
    upload_id: str,
    analysis_id: str,
):
    """Inserts a review into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO user_reviews (review_content, flagged, confidence, upload_id, analysis_id)
            VALUES (?, ?, ?, ?, ?)
        """,
            (review_content, flagged, confidence, upload_id, analysis_id),
        )
        conn.commit()
        logger.info("Review inserted successfully.")
        return cursor.lastrowid  # Return the ID of the new row
    except sqlite3.Error as e:
        logger.error(f"Error inserting review: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


def get_reviews_by_analysis_id(analysis_id: str) -> list:
    """Retrieves reviews associated with a specific analysis ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT * FROM user_reviews WHERE analysis_id = ?
        """,
            (analysis_id,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]  # Convert rows to dictionaries
    except sqlite3.Error as e:
        logger.error(f"Error retrieving reviews: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def insert_analysis_job(job_data: Dict[str, Any]):
    """Inserts a new analysis job into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO analysis_jobs (
                analysis_id, file_name, status, progress, result_url,
                error_message, updated_at, total_reviews, flagged_reviews,
                upload_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                job_data["analysis_id"],
                job_data.get("file_name"),
                job_data.get("status"),
                job_data.get("progress"),
                job_data.get("result_url"),
                job_data.get("error_message"),
                job_data.get("updated_at"),
                job_data.get("total_reviews"),
                job_data.get("flagged_reviews"),
                job_data.get("upload_id"),
            ),
        )
        conn.commit()
        logger.info(f"Analysis job {job_data['analysis_id']} inserted successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error inserting analysis job {job_data['analysis_id']}: {e}")
    finally:
        cursor.close()
        conn.close()


def get_analysis_job_by_id(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves an analysis job from the database by its ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT * FROM analysis_jobs WHERE analysis_id = ?
        """,
            (analysis_id,),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)  # Convert to dictionary
        else:
            return None
    except sqlite3.Error as e:
        logger.error(f"Error retrieving analysis job {analysis_id}: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


def update_analysis_job(analysis_id: str, job_data: Dict[str, Any]):
    """Updates an existing analysis job in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Construct the SET clause dynamically
        set_clause = ", ".join(
            f"{key} = ?" for key in job_data if key != "analysis_id"
        )  # Exclude analysis_id from update
        if not set_clause:
            logger.warning(f"No data to update for analysis job {analysis_id}.")
            return  # Nothing to update

        cursor.execute(
            f"""
            UPDATE analysis_jobs
            SET {set_clause}
            WHERE analysis_id = ?
        """,
            tuple(job_data[key] for key in job_data if key != "analysis_id")
            + (analysis_id,),  # Add analysis_id at the end for WHERE clause
        )
        conn.commit()
        if cursor.rowcount > 0:
            logger.info(f"Analysis job {analysis_id} updated successfully.")
        else:
            logger.warning(f"Analysis job {analysis_id} not found for update.")
    except sqlite3.Error as e:
        logger.error(f"Error updating analysis job {analysis_id}: {e}")
    finally:
        cursor.close()
        conn.close()


def delete_analysis_job(analysis_id: str):
    """Deletes an analysis job from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            DELETE FROM analysis_jobs WHERE analysis_id = ?
        """,
            (analysis_id,),
        )
        conn.commit()
        if cursor.rowcount > 0:
            logger.info(f"Analysis job {analysis_id} deleted successfully.")
        else:
            logger.warning(f"Analysis job {analysis_id} not found for deletion.")
    except sqlite3.Error as e:
        logger.error(f"Error deleting analysis job {analysis_id}: {e}")
    finally:
        cursor.close()
        conn.close()


def get_all_analysis_jobs_from_db() -> List[Dict[str, Any]]:
    """Retrieves all analysis jobs from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT * FROM analysis_jobs
        """
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]  # Convert rows to dictionaries
    except sqlite3.Error as e:
        logger.error(f"Error retrieving all analysis jobs: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


# Call create_tables when the module is imported
try:
    create_tables()
except Exception as e:
    logger.critical(f"Failed to initialize database: {e}")
    # Consider exiting the application if the database is critical
    raise
