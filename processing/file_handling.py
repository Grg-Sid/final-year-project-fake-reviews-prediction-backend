# processing/file_handling.py
import PyPDF2
from typing import List
from core.config import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """Extract text from PDF and split into potential reviews."""
    try:
        text_content = []
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            logger.info(f"Reading PDF '{pdf_path}' with {num_pages} pages.")
            for page_num in range(num_pages):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:  # Only append if text was extracted
                        text_content.append(page_text)
                    else:
                        logger.warning(
                            f"No text extracted from page {page_num + 1} of '{pdf_path}'."
                        )
                except Exception as page_e:
                    logger.error(
                        f"Error extracting text from page {page_num + 1} of '{pdf_path}': {page_e}"
                    )

        # Simple parsing: split by newlines and filter out short/empty lines
        reviews = []
        for page_text in text_content:
            lines = page_text.split("\n")
            for line in lines:
                cleaned_line = line.strip()
                # Adjust filter logic as needed (e.g., minimum word count)
                if len(cleaned_line) > 20:  # Assuming reviews are longer than 20 chars
                    reviews.append(cleaned_line)

        logger.info(f"Extracted {len(reviews)} potential reviews from '{pdf_path}'.")
        return reviews
    except FileNotFoundError:
        logger.error(f"PDF file not found at path: {pdf_path}")
        return []
    except Exception as e:
        logger.error(f"Error extracting text from PDF '{pdf_path}': {str(e)}")
        return []
