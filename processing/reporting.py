# processing/reporting.py
from fpdf import FPDF
from typing import List, Dict, Any
from datetime import datetime
from core.config import get_logger
import os

logger = get_logger(__name__)


def generate_pdf_report(
    analysis_results: List[Dict[str, Any]], output_path: str
) -> bool:
    """Generate a PDF report from analysis results."""
    try:
        pdf = FPDF()
        pdf.add_page()

        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Review Analysis Report", ln=True, align="C")
        pdf.ln(10)

        # Summary
        total_reviews = len(analysis_results)
        flagged_reviews = sum(1 for r in analysis_results if r["flagged"])
        flagged_percentage = (
            (flagged_reviews / total_reviews) * 100 if total_reviews > 0 else 0
        )

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Summary:", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 8, f"Total Reviews Analyzed: {total_reviews}", ln=True)
        pdf.cell(
            0,
            8,
            f"Flagged Reviews: {flagged_reviews} ({flagged_percentage:.1f}%)",
            ln=True,
        )
        pdf.cell(
            0,
            8,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ln=True,
        )
        pdf.ln(10)

        # Detailed Results
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Detailed Results:", ln=True)

        # Table Headers
        pdf.set_font("Arial", "B", 10)
        pdf.cell(10, 10, "#", border=1)
        pdf.cell(120, 10, "Review Content (Truncated)", border=1)
        pdf.cell(30, 10, "Flagged", border=1)
        pdf.cell(30, 10, "Confidence", border=1, ln=True)

        # Table Content
        pdf.set_font("Arial", "", 9)
        for i, result in enumerate(analysis_results, 1):
            # Truncate review content to fit in the cell
            review_content = result["review_content"]
            if len(review_content) > 70:
                review_content = review_content[:67] + "..."

            # Row content
            pdf.cell(10, 10, str(i), border=1)
            pdf.cell(120, 10, review_content, border=1)
            pdf.cell(30, 10, "Yes" if result["flagged"] else "No", border=1)
            pdf.cell(30, 10, f"{result['confidence']:.2f}", border=1, ln=True)

            # If we have too many entries, limit them
            if i >= 100:
                pdf.cell(
                    0,
                    10,
                    f"... and {len(analysis_results) - 100} more reviews (truncated)",
                    ln=True,
                )
                break

        # Save PDF
        pdf.output(output_path)
        return True
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        return False


def generate_pdf_report_2(
    analysis_results: List[Dict[str, Any]], output_path: str
) -> bool:
    """Generate a PDF report from analysis results."""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Review Analysis Report", ln=True, align="C")
        pdf.ln(10)

        # Summary
        total_reviews = len(analysis_results)
        if total_reviews == 0:
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, "No reviews were processed.", ln=True)
            pdf.output(output_path)
            logger.info(f"Generated empty PDF report (no results): {output_path}")
            return True

        flagged_reviews = sum(1 for r in analysis_results if r.get("flagged", False))
        flagged_percentage = (
            (flagged_reviews / total_reviews) * 100 if total_reviews > 0 else 0
        )

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Summary:", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 8, f"Total Reviews Analyzed: {total_reviews}", ln=True)
        pdf.cell(
            0,
            8,
            f"Flagged Reviews: {flagged_reviews} ({flagged_percentage:.1f}%)",
            ln=True,
        )
        pdf.cell(
            0,
            8,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ln=True,
        )
        pdf.ln(10)

        # Detailed Results - Limit to avoid huge PDFs
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Detailed Results (Sample):", ln=True)

        # Table Headers
        pdf.set_font("Arial", "B", 10)
        col_widths = {
            "#": 10,
            "Content": 110,
            "Flagged": 25,
            "Confidence": 25,
        }  # Adjust widths as needed
        pdf.cell(col_widths["#"], 10, "#", border=1, align="C")
        pdf.cell(
            col_widths["Content"], 10, "Review Content (Truncated)", border=1, align="C"
        )
        pdf.cell(col_widths["Flagged"], 10, "Flagged", border=1, align="C")
        pdf.cell(
            col_widths["Confidence"], 10, "Confidence", border=1, align="C", ln=True
        )

        # Table Content
        pdf.set_font("Arial", "", 9)
        max_results_in_pdf = 100  # Limit results shown in PDF
        for i, result in enumerate(analysis_results[:max_results_in_pdf], 1):
            # Handle potential errors in results gracefully
            review_content = result.get("review_content", "N/A")
            flagged = result.get("flagged", False)
            confidence = result.get("confidence", 0.0)

            # Truncate review content safely
            max_content_len = 70
            if len(review_content) > max_content_len:
                review_content_display = review_content[: max_content_len - 3] + "..."
            else:
                review_content_display = review_content

            # Add row cells
            # Use multi_cell for review content to allow wrapping
            start_x = pdf.get_x()
            start_y = pdf.get_y()
            pdf.multi_cell(col_widths["#"], 10, str(i), border=1, align="C")
            current_y = pdf.get_y()
            pdf.set_xy(start_x + col_widths["#"], start_y)  # Reset X position

            pdf.multi_cell(
                col_widths["Content"],
                10,
                review_content_display.encode("latin-1", "replace").decode("latin-1"),
                border=1,
            )  # Handle potential encoding issues for FPDF
            content_cell_height = pdf.get_y() - start_y  # Height of the content cell
            pdf.set_xy(
                start_x + col_widths["#"] + col_widths["Content"], start_y
            )  # Reset X position

            pdf.multi_cell(
                col_widths["Flagged"],
                content_cell_height,
                "Yes" if flagged else "No",
                border=1,
                align="C",
            )
            pdf.set_xy(
                start_x
                + col_widths["#"]
                + col_widths["Content"]
                + col_widths["Flagged"],
                start_y,
            )  # Reset X position

            pdf.multi_cell(
                col_widths["Confidence"],
                content_cell_height,
                f"{confidence:.3f}",
                border=1,
                align="C",
            )

        if total_reviews > max_results_in_pdf:
            pdf.set_font("Arial", "I", 10)
            pdf.cell(
                0,
                10,
                f"... and {total_reviews - max_results_in_pdf} more reviews (results available in CSV).",
                ln=True,
            )

        # Save PDF
        pdf.output(output_path)
        logger.info(f"Successfully generated PDF report: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error generating PDF report '{output_path}': {str(e)}")
        # Attempt to remove partial file if error occurred
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass  # Ignore error during cleanup
        return False
