"""
Specialized seed packet processor that uses the image processor for OCR and data extraction.
Handles the specific workflow for extracting seed data from packet images.
"""

import os
import logging
import datetime
import uuid
import json
import time
from typing import Tuple, Dict, Any, Optional

# Import the existing image processor
from utils.image_processor import image_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("seed_packet_processor")


class SeedPacketProcessor:
    """Processor for seed packet images, extracting structured data for seeds."""

    def __init__(self):
        # Use the upload_dir from image_processor to maintain consistency
        self.upload_dir = image_processor.upload_dir
        logger.info(
            f"Seed packet processor initialized with upload dir: {self.upload_dir}"
        )

        # Log the configured vision API provider
        self.vision_api_provider = os.getenv("VISION_API_PROVIDER", "claude").lower()
        logger.info(f"Using vision API provider: {self.vision_api_provider}")

        # Check for API keys and log status
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")

        if self.vision_api_provider == "gemini" and not self.gemini_api_key:
            logger.warning("Gemini selected as provider but GEMINI_API_KEY not found!")
        elif self.vision_api_provider == "claude" and not self.claude_api_key:
            logger.warning(
                "Claude selected as provider but ANTHROPIC_API_KEY not found!"
            )
        else:
            logger.info(f"API key for {self.vision_api_provider} is configured")

    async def process_seed_packet(
        self, image_data: bytes, filename: str
    ) -> Tuple[Dict[str, Any], str]:
        """
        Process a seed packet image to extract structured data.

        Args:
            image_data: The binary image data
            filename: Original filename of the uploaded file

        Returns:
            Tuple of (structured_data, file_path)
        """
        try:
            logger.info(f"Processing seed packet image: {filename}")
            total_start = time.perf_counter()

            # Generate a unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            file_ext = os.path.splitext(filename)[1].lower() if filename else ".jpg"
            new_filename = f"seed_{timestamp}_{unique_id}{file_ext}"
            file_path = os.path.join(self.upload_dir, new_filename)

            # Save the image to disk
            save_start = time.perf_counter()
            with open(file_path, "wb") as f:
                f.write(image_data)
            save_end = time.perf_counter()
            logger.info(
                f"Saved image to {file_path} (save time: {save_end-save_start:.2f}s)"
            )

            # Preprocessing (if any) is handled in image_processor
            vision_start = time.perf_counter()
            try:
                _, structured_data = (
                    await image_processor.process_image_with_vision_api(file_path)
                )
                vision_end = time.perf_counter()
                logger.info(
                    f"Structured data extracted (Gemini API call time: {vision_end-vision_start:.2f}s)"
                )
                logger.info(f"Structured data: {json.dumps(structured_data)[:200]}...")
            except Exception as e:
                vision_end = time.perf_counter()
                logger.error(
                    f"Error during vision API processing: {str(e)} (Gemini API call time: {vision_end-vision_start:.2f}s)"
                )
                structured_data = None

            if not structured_data:
                logger.warning("No structured data obtained, creating default")
                structured_data = {
                    "name": "Unknown Seed",
                    "variety": None,
                    "notes": "No data could be extracted automatically. Please fill in the details manually.",
                }

            # Ensure all required fields exist with defaults if needed
            structured_data = self._process_structured_data(structured_data)

            total_end = time.perf_counter()
            logger.info(
                f"Total seed packet processing time: {total_end-total_start:.2f}s"
            )

            return structured_data, file_path

        except Exception as e:
            logger.error(f"Error processing seed packet: {str(e)}", exc_info=True)
            # Return a fallback response so the app doesn't crash
            structured_data = {
                "name": "Unknown Seed",
                "variety": None,
                "brand": None,
                "germination_rate": None,
                "maturity": None,
                "growth": None,
                "seed_depth": None,
                "spacing": None,
                "notes": f"Error during processing: {str(e)}. Please enter details manually.",
            }
            return structured_data, file_path if "file_path" in locals() else ""

    def _process_structured_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure all necessary fields exist in the structured data and have appropriate types.
        Convert fields as needed to match the Seed model.
        """
        # Create a new dict with all required fields and proper defaults
        processed = {
            "name": data.get("name", "Unknown Seed"),
            "variety": data.get("variety"),
            "brand": data.get("brand"),
            "germination_rate": None,
            "maturity": None,
            "growth": data.get("growth"),
            "seed_depth": None,
            "spacing": None,
            "notes": data.get("notes", ""),
        }

        # Process numeric fields with appropriate type conversion
        if "germination_rate" in data and data["germination_rate"] is not None:
            try:
                # Ensure germination rate is a float between 0 and 1
                rate = float(data["germination_rate"])
                if rate > 1:  # If it's a percentage (like 85), convert to decimal
                    rate = rate / 100
                processed["germination_rate"] = max(
                    0, min(1, rate)
                )  # Clamp between 0 and 1
            except (ValueError, TypeError):
                logger.warning("Could not convert germination_rate to float")

        if "maturity" in data and data["maturity"] is not None:
            try:
                processed["maturity"] = int(data["maturity"])
            except (ValueError, TypeError):
                logger.warning("Could not convert maturity to integer")

        if "seed_depth" in data and data["seed_depth"] is not None:
            try:
                processed["seed_depth"] = float(data["seed_depth"])
            except (ValueError, TypeError):
                logger.warning("Could not convert seed_depth to float")

        if "spacing" in data and data["spacing"] is not None:
            try:
                processed["spacing"] = float(data["spacing"])
            except (ValueError, TypeError):
                logger.warning("Could not convert spacing to float")

        return processed


# Create a singleton instance for use throughout the app
seed_packet_processor = SeedPacketProcessor()
