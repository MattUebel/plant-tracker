"""
Client-side script to bulk import seed packet images.

This script iterates through image files in a specified directory on the host machine
and sends them to the Plant Tracker application's bulk import API endpoint for processing.

Usage:
    export PLANT_TRACKER_URL="http://localhost:8000" # Or your app's URL
    python scripts/bulk_import_seeds.py /path/to/your/seed/images

Requires the 'requests' library: pip install requests
"""

import os
import sys
import requests
import argparse
import logging
from pathlib import Path
from mimetypes import guess_type
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("bulk_import_client")

# --- Configuration --- #
APP_URL = os.getenv("PLANT_TRACKER_URL")
ENDPOINT = "/bulk-import/process-image"

# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def process_image(
    image_path: Path,
    base_url: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
):
    """Sends a single image to the bulk import API endpoint."""
    url = f"{base_url.rstrip('/')}{ENDPOINT}"
    params = {}
    if provider:
        params["provider"] = provider
    if model:
        params["model"] = model

    mime_type, _ = guess_type(image_path)
    if not mime_type or not mime_type.startswith("image/"):
        logger.warning(f"Skipping non-image file: {image_path.name}")
        return False, "Skipped (not an image)"

    try:
        with open(image_path, "rb") as f:
            files = {"image_file": (image_path.name, f, mime_type)}
            response = requests.post(
                url, files=files, params=params, timeout=120
            )  # Increased timeout, added params

        if response.status_code == 200:
            logger.info(
                f"Successfully processed: {image_path.name} -> Seed ID: {response.json().get('seed_id')}"
            )
            return True, response.json()
        else:
            error_detail = response.text
            try:
                error_detail = response.json().get("detail", response.text)
            except requests.exceptions.JSONDecodeError:
                pass  # Keep the raw text if it's not JSON
            logger.error(
                f"Failed to process {image_path.name}: {response.status_code} - {error_detail}"
            )
            return False, f"Error {response.status_code}: {error_detail}"

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error processing {image_path.name}: {e}")
        return False, f"Network error: {e}"
    except Exception as e:
        logger.error(
            f"Unexpected error processing {image_path.name}: {e}", exc_info=True
        )
        return False, f"Unexpected error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Bulk import seed packet images into Plant Tracker."
    )
    parser.add_argument(
        "image_directory", type=str, help="Directory containing seed packet images."
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Vision API provider to use (e.g., claude, gemini, mistral). Overrides server default.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model name to use. Overrides server default for the chosen provider.",
    )
    args = parser.parse_args()

    if not APP_URL:
        logger.critical("PLANT_TRACKER_URL environment variable is not set.")
        sys.exit(1)

    image_dir = Path(args.image_directory)
    if not image_dir.is_dir():
        logger.critical(f"Error: Directory not found - {args.image_directory}")
        sys.exit(1)

    logger.info(f"Starting bulk import from directory: {image_dir}")
    logger.info(f"Target application URL: {APP_URL}")

    success_count = 0
    failure_count = 0
    skipped_count = 0
    total_files = 0

    for item in image_dir.iterdir():
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            total_files += 1
            logger.info(f"Processing file: {item.name}...")
            success, result = process_image(
                item, APP_URL, args.provider, args.model
            )  # Pass args
            if success:
                success_count += 1
            else:
                if "Skipped" in str(result):
                    skipped_count += 1
                else:
                    failure_count += 1
        elif item.is_file():
            logger.debug(f"Skipping non-supported file type: {item.name}")
            skipped_count += 1
            total_files += 1  # Count as processed for summary

    logger.info("--- Bulk Import Summary ---")
    logger.info(f"Total files found: {total_files}")
    logger.info(f"Successfully imported: {success_count}")
    logger.info(f"Failed imports: {failure_count}")
    logger.info(f"Skipped files: {skipped_count}")
    logger.info("Bulk import process finished.")


if __name__ == "__main__":
    main()
