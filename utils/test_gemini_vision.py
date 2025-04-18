#!/usr/bin/env python3
"""
Advanced testing utility for Google's Gemini Vision API with seed packet images.
This script provides enhanced features for testing the Gemini API with seed packet images,
including the ability to test specific images, save results to files, and compare
results between different models.

Usage:
    cd /Users/mattuebel/mattuebel/plant-tracker
    python -m utils.test_gemini_vision [options]

Examples:
    # Test all images in the uploads directory with default model
    python -m utils.test_gemini_vision

    # Test with a specific model
    python -m utils.test_gemini_vision --model gemini-2.0-pro-latest

    # Test a specific image or set of images
    python -m utils.test_gemini_vision --image uploads/seed_1_20250318005632_ce2b23dd.JPG

    # Save results to output files
    python -m utils.test_gemini_vision --save-dir ./test_results

    # Verbose output with timing information
    python -m utils.test_gemini_vision --verbose
"""
import os
import sys
import base64
import asyncio
import json
import time
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from PIL import Image
from io import BytesIO

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Load environment variables from project root
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Import the Gemini Vision API implementation
from utils.gemini_vision_api import ImageProcessor, GeminiVisionTester

# Check API key setup
if not os.getenv("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY not found in environment variables.")
    print(
        "Please set your GEMINI_API_KEY in your .env file or export it in your environment."
    )
    sys.exit(1)

# Lazy import to avoid dependency issues if not using Gemini
try:
    # Import the new google-genai library
    import genai
except ImportError:
    print("Error: google-genai package is not installed.")
    print("Please install it using: pip install google-genai")
    sys.exit(1)


class GeminiVisionTester(GeminiVisionTester):
    """Enhanced Gemini Vision API tester with additional features."""

    async def extract_structured_data_with_json_mode(
        self, image_path: str
    ) -> Dict[str, Any]:
        """
        Extract structured data using JSON response format mode.
        This is an alternative approach that may produce cleaner JSON results.

        Args:
            image_path: Path to the image file

        Returns:
            Dict containing the structured seed packet data
        """
        prompt = """
        Extract structured plant information from this seed packet image.
        Return a valid JSON object matching the following fields and types:
        {
            "name": "The main name/type of the plant (e.g., Tomato, Basil) - String",
            "variety": "The specific variety of the plant/seed - String",
            "brand": "The brand or company that produced the seed packet - String",
            "germination_rate": "The germination rate as a decimal between 0.0-1.0 (e.g., 0.85 for 85%) - Float",
            "maturity": "Days to maturity/harvest as a number - Integer",
            "growth": "Growth habit or pattern (e.g., Determinate, Bush, Vining) - String",
            "seed_depth": "Recommended planting depth in inches - Float",
            "spacing": "Recommended spacing between plants in inches - Float",
            "quantity": "Number of seeds in the packet if mentioned - Integer",
            "notes": "Additional important information from the packet - String"
        }
        
        For the germination rate, if it's provided as a percentage (e.g., 85%), convert it to decimal (0.85).
        For seed_depth and spacing, convert to inches if given in other units.
        If any information is not available in the image, use null for that field.
        """

        # Process the image to ensure it's under size limit
        try:
            image_bytes = self.image_processor.resize_image_for_gemini(image_path)
            img = Image.open(BytesIO(image_bytes))
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}

        try:
            # Preparing content parts (text prompt + image)
            contents = [prompt, img]

            # Configure response format for JSON
            generation_config = {"response_format": "json", "temperature": 0.2}

            # Make the API call
            response = await asyncio.to_thread(
                self.model.generate_content,
                contents,
                generation_config=generation_config,
            )

            if hasattr(response, "text"):
                try:
                    # Should be able to parse directly as the response is constrained to JSON
                    return json.loads(response.text)
                except json.JSONDecodeError as e:
                    return {
                        "error": f"Failed to parse JSON response: {str(e)}",
                        "raw_response": response.text,
                    }
            else:
                return {
                    "error": "Failed to extract structured data",
                    "response": str(response),
                }

        except Exception as e:
            return {"error": f"Error calling Gemini API for structured data: {str(e)}"}


class TestRunner:
    """Test runner for Gemini Vision API tests."""

    def __init__(self, args):
        self.args = args
        self.model_name = args.model
        self.verbose = args.verbose
        self.save_dir = args.save_dir
        self.image_paths = self._get_image_paths()

        # Create save directory if saving results
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Results will be saved to: {self.save_dir}")

    def _get_image_paths(self) -> List[Path]:
        """Get the list of image paths to process based on command line arguments."""
        if self.args.image:
            # Process specific images provided as arguments
            paths = []
            for img_path in self.args.image:
                path = Path(img_path)
                if path.exists() and path.is_file():
                    paths.append(path)
                else:
                    print(f"Warning: Image file not found: {img_path}")
            return paths
        else:
            # Process all images in the uploads directory
            uploads_dir = Path(self.args.uploads_dir)
            if not uploads_dir.exists():
                print(f"Error: Uploads directory not found: {uploads_dir}")
                sys.exit(1)

            # Find all image files
            image_files = []
            for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
                image_files.extend([f for f in uploads_dir.glob(ext) if f.is_file()])

            return image_files

    async def run_tests(self):
        """Run the Gemini Vision API tests on the specified images."""
        if not self.image_paths:
            print("No images found to process.")
            return

        print(f"Using Gemini model: {self.model_name}")
        print(f"Found {len(self.image_paths)} image(s) to process.")

        # Initialize the Gemini Vision tester
        vision_tester = GeminiVisionTester(self.model_name)

        # Create a timestamp for saving results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Process each image
        for i, image_path in enumerate(self.image_paths):
            print(f"\n{'='*60}")
            print(f"Processing image {i+1}/{len(self.image_paths)}: {image_path.name}")
            print(f"{'='*60}")

            results = {}

            # Extract OCR text
            print("\n1. Extracting raw OCR text using Gemini Vision API...")
            start_time = time.time()
            try:
                ocr_result = await vision_tester.extract_ocr_text(str(image_path))

                if "error" not in ocr_result:
                    elapsed = time.time() - start_time
                    if self.verbose:
                        print(f"OCR extraction took: {elapsed:.2f} seconds")

                    extracted_text = ocr_result["text"]
                    results["ocr_text"] = extracted_text

                    print("\nExtracted OCR text:")
                    print("-" * 50)
                    # Print first 500 chars with ellipsis if longer
                    print(
                        extracted_text[:500]
                        + ("..." if len(extracted_text) > 500 else "")
                    )
                    print("-" * 50)
                else:
                    print(f"\nError extracting OCR text: {ocr_result['error']}")
                    results["ocr_error"] = ocr_result["error"]
            except Exception as e:
                print(f"\nException during OCR extraction: {str(e)}")
                results["ocr_exception"] = str(e)

            # Extract structured data
            print("\n2. Extracting structured data using Gemini Vision API...")
            start_time = time.time()
            try:
                # Try using the standard method first
                structured_data = await vision_tester.extract_structured_data(
                    str(image_path)
                )

                # If there's an error, try the JSON mode method (if enabled)
                if "error" in structured_data and self.args.use_json_mode:
                    print("\nRetrying with JSON response mode...")
                    structured_data = (
                        await vision_tester.extract_structured_data_with_json_mode(
                            str(image_path)
                        )
                    )

                if "error" not in structured_data:
                    elapsed = time.time() - start_time
                    if self.verbose:
                        print(f"Structured data extraction took: {elapsed:.2f} seconds")

                    results["structured_data"] = structured_data

                    print("\nExtracted structured data:")
                    print("-" * 50)
                    print(json.dumps(structured_data, indent=2))
                    print("-" * 50)

                    # Print data validation information
                    missing_fields = [
                        field
                        for field in [
                            "name",
                            "variety",
                            "maturity",
                            "seed_depth",
                            "spacing",
                        ]
                        if field not in structured_data
                        or structured_data[field] is None
                    ]
                    if missing_fields:
                        print("\nMissing important fields:", ", ".join(missing_fields))
                        results["missing_fields"] = missing_fields
                    else:
                        print("\nAll key fields were extracted successfully.")
                        results["missing_fields"] = []
                else:
                    print(
                        f"\nError extracting structured data: {structured_data['error']}"
                    )
                    results["structured_data_error"] = structured_data["error"]
                    if "raw_response" in structured_data:
                        print("\nRaw response from Gemini:")
                        raw_response = structured_data["raw_response"]
                        print(
                            raw_response[:500] + "..."
                            if len(raw_response) > 500
                            else raw_response
                        )
                        results["raw_response"] = raw_response
            except Exception as e:
                print(f"\nException during structured data extraction: {str(e)}")
                results["structured_data_exception"] = str(e)

            # Save results to file if requested
            if self.save_dir:
                result_filename = f"{timestamp}_{image_path.stem}_gemini_results.json"
                result_path = Path(self.save_dir) / result_filename

                with open(result_path, "w") as f:
                    json.dump(
                        {
                            "image_filename": image_path.name,
                            "model": self.model_name,
                            "timestamp": timestamp,
                            "results": results,
                        },
                        f,
                        indent=2,
                    )

                print(f"\nResults saved to: {result_path}")

            # Wait between images to avoid rate limits
            if i < len(self.image_paths) - 1:
                wait_time = 2
                print(f"\nWaiting {wait_time} seconds before processing next image...")
                await asyncio.sleep(wait_time)


async def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Test Google Gemini Vision API with seed packet images."
    )
    parser.add_argument(
        "image_path", type=str, help="Path to the seed packet image file."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-pro-preview-03-25"),
        help="Gemini model name to use (default: gemini-2.5-pro-preview-03-25 or GEMINI_MODEL from .env)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Directory to save output files."
    )
    parser.add_argument(
        "--image",
        nargs="+",
        help="Path(s) to specific image(s) to process (if not specified, all images in uploads directory will be processed)",
    )
    parser.add_argument(
        "--uploads-dir",
        default=str(Path(__file__).parent.parent / "uploads"),
        help="Directory containing images to process (default: uploads directory)",
    )
    parser.add_argument(
        "--save-dir",
        help="Directory to save results (if not specified, results will not be saved to files)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with additional details like timing information",
    )
    parser.add_argument(
        "--use-json-mode",
        action="store_true",
        help="Try using JSON response mode if standard extraction fails",
    )

    args = parser.parse_args()

    # Run the tests
    runner = TestRunner(args)
    await runner.run_tests()


if __name__ == "__main__":
    asyncio.run(main())
