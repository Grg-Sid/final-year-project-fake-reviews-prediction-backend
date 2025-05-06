# utils/cleanup.py
import os
import time
from core.config import UPLOAD_DIR, RESULTS_DIR, CLEANUP_FILE_AGE_SECONDS, get_logger

logger = get_logger(__name__)

def cleanup_old_files():
    """Removes files older than a configured age from upload and result directories."""
    logger.info("Running cleanup routine for old files...")
    cleaned_count = 0
    error_count = 0
    for directory in [UPLOAD_DIR, RESULTS_DIR]:
        if not os.path.exists(directory):
            logger.warning(f"Cleanup directory not found: {directory}")
            continue

        try:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    # Check if it's a file and if it's old enough
                    if (
                        os.path.isfile(file_path)
                        and time.time() - os.path.getmtime(file_path) > CLEANUP_FILE_AGE_SECONDS
                    ):
                        os.remove(file_path)
                        logger.info(f"Removed old file: {file_path}")
                        cleaned_count += 1
                except FileNotFoundError:
                    continue # File might have been deleted between listdir and check
                except Exception as e:
                    logger.error(f"Error removing file {file_path}: {str(e)}")
                    error_count += 1
        except Exception as list_e:
             logger.error(f"Error listing directory {directory} for cleanup: {list_e}")
             error_count += 1

    logger.info(f"Cleanup finished. Removed {cleaned_count} files. Encountered {error_count} errors.")