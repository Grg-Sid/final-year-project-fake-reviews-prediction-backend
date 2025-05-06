# processing/nlp.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from core.config import get_logger

logger = get_logger(__name__)

# --- Download necessary NLTK resources ---
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    # nltk.download("punkt_tab", quiet=True) # Often included with 'punkt'
    stopwords_set = set(stopwords.words("english"))
    logger.info("NLTK resources downloaded successfully.")
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")
    raise RuntimeError(f"Failed to download NLTK resources: {e}")


# --- Text Processing Function ---
def process_text(text: str) -> str:
    """Cleans and tokenizes text for vectorization."""
    if not isinstance(text, str):
        return ""
    try:
        # Ensure text is treated as string and handle potential non-string types gracefully
        tokens = word_tokenize(str(text).lower())
        return " ".join(
            [word for word in tokens if word.isalnum() and word not in stopwords_set]
        )
    except Exception as e:
        logger.error(f"Error processing text: '{text[:50]}...': {str(e)}")
        return ""  # Return empty string on error to avoid downstream issues
