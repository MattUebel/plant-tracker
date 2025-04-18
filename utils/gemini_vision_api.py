"""
Utility script to use Google's Gemini API with seed packet images.
This script extracts both raw OCR text and structured data from seed packet images
based on the application's Seed model schema.
Usage:
    cd /Users/mattuebel/mattuebel/plant-tracker
    python -m utils.gemini_vision_api
"""

import os
import sys
import base64
import asyncio
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
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

# Check API key setup
if not os.getenv("GEMINI_API_KEY"):
    print("Warning: GEMINI_API_KEY not found in environment variables.")
    print("Please set your GEMINI_API_KEY using:")
    print("export GEMINI_API_KEY='your_api_key_here'")
    print("Or add it to your .env file.")
    sys.exit(1)

# Import Google's Generative AI library
try:
    import google.generativeai as genai

    print("Successfully imported google.generativeai")
except ImportError as e:
    print(f"Error importing Google Generative AI library: {e}")
    print("Please install with: python -m pip install google-generativeai")
    sys.exit(1)


class ImageProcessor:
    """Image processing utilities to prepare images for Gemini API."""

    @staticmethod
    def resize_image_for_gemini(
        image_path: str, max_size_bytes: int = 4.5 * 1024 * 1024
    ) -> bytes:
        """
        Resize and compress an image to ensure it's under Gemini's size limit.

        Args:
            image_path: Path to the image file
            max_size_bytes: Maximum size in bytes (default: 4.5MB to have safety margin)

        Returns:
            Processed image as bytes
        """
        # Open the image with PIL for better quality control
        with Image.open(image_path) as img:
            # Convert to RGB if needed (removes alpha channel)
            if img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            ):
                img = img.convert("RGB")

            # Start with original dimensions
            width, height = img.size
            quality = 90

            # Create a BytesIO object to check size without saving to disk
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="JPEG", quality=quality)
            current_size = img_byte_arr.tell()

            # Iteratively reduce size until under limit
            while current_size > max_size_bytes:
                if quality > 50:
                    # First try reducing quality
                    quality -= 10
                else:
                    # Then try reducing dimensions
                    width = int(width * 0.9)
                    height = int(height * 0.9)
                    img = img.resize((width, height), Image.Resampling.LANCZOS)

                # Check new size
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format="JPEG", quality=quality)
                current_size = img_byte_arr.tell()

            # Get the processed image as bytes
            img_byte_arr.seek(0)
            return img_byte_arr.getvalue()


class GeminiAPICaller:
    """Base class for Google Gemini API calls with retry logic."""

    def __init__(self, model_name: str = "gemini-2.5-pro-preview-03-25"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        # Initialize the Google AI Client using google-generativeai
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.max_retries = 3
        self.initial_retry_delay = 1.0
        self.image_processor = ImageProcessor()

    async def call_api_with_retry(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Call Gemini API with an image and prompt using exponential backoff retry logic.

        Args:
            image_path: Path to the image file
            prompt: The prompt to send to Gemini

        Returns:
            Gemini API response
        """
        retries = 0
        delay = self.initial_retry_delay
        last_exception = None

        # Process the image to ensure it's under size limit
        try:
            print(f"Processing image to fit within size limits...")
            image_bytes = self.image_processor.resize_image_for_gemini(image_path)
            image_size_mb = len(image_bytes) / (1024 * 1024)
            print(f"Image processed. New size: {image_size_mb:.2f} MB")

            # Load the image for the Gemini API
            img = Image.open(BytesIO(image_bytes))
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise

        while retries <= self.max_retries:
            try:
                # Preparing content parts (text prompt + image)
                response = await asyncio.to_thread(
                    self.model.generate_content, [prompt, img]
                )

                return response

            except Exception as e:
                print(f"API call failed: {str(e)}")
                if retries < self.max_retries:
                    print(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                    retries += 1
                    last_exception = e
                else:
                    if last_exception:
                        raise last_exception
                    raise

        if last_exception:
            raise last_exception

        return {"error": "Max retries exceeded"}


class GeminiVisionTester(GeminiAPICaller):
    """Test Gemini Vision API for image understanding and data extraction."""

    def __init__(self, model_name: str = "gemini-2.5-pro-preview-03-25"):
        super().__init__(model_name)

    async def extract_ocr_text(self, image_path: str) -> Dict[str, Any]:
        """
        Extract pure OCR text from an image using Gemini Vision API.

        Args:
            image_path: Path to the image file

        Returns:
            Dict containing the raw extracted text
        """
        prompt = """
        Perform OCR on this seed packet image. 
        Extract ALL text visible in the image, preserving the layout as much as possible.
        Include all product information, instructions, and details exactly as they appear.
        Focus only on the text content - do not analyze or interpret the information.
        """

        response = await self.call_api_with_retry(image_path, prompt)

        if hasattr(response, "text"):
            # Extract text from response
            text = response.text
            return {"text": text}
        else:
            return {"error": "Failed to extract OCR text", "response": str(response)}

    async def extract_structured_data(
        self, image_path: str, genai_client=None
    ) -> Dict[str, Any]:
        """
        Extract structured seed packet data from an image using Gemini Vision API.
        The data structure matches the application's Seed model schema.

        Args:
            image_path: Path to the image file

        Returns:
            Dict containing the structured seed packet data
        """
        prompt = """
        Analyze this seed packet image and extract the following information as a JSON object:
        - name: The main plant type (e.g., 'Tomato', 'Basil', 'Carrot')
        - variety: The specific variety name (e.g., 'Cherry Sweet', 'Genovese', 'Nantes')
        - brand: The company/manufacturer of the seed packet
        - seed_depth: Recommended planting depth in inches (convert from other units if needed)
        - spacing: Recommended spacing between plants in inches (convert from other units if needed)
        - notes: Any special growing instructions or other important information

        Return ONLY a valid JSON object with these fields. Use null for any fields not found in the image.
        """

        response = await self.call_api_with_retry(image_path, prompt)

        if hasattr(response, "text"):
            # Extract JSON from response
            content_text = response.text
            try:
                # Find JSON object in the response text (Gemini might add text around the JSON)
                json_str = content_text
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()

                # If there's still no clear JSON, try looking for { } brackets
                if "{" in json_str and "}" in json_str:
                    start_idx = json_str.find("{")
                    end_idx = json_str.rfind("}") + 1
                    json_str = json_str[start_idx:end_idx]

                structured_data = json.loads(json_str)
                return structured_data
            except json.JSONDecodeError as e:
                return {
                    "error": f"Failed to parse JSON response: {str(e)}",
                    "raw_response": content_text,
                }
        else:
            return {
                "error": "Failed to extract structured data",
                "response": str(response),
            }

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
        I need you to analyze this seed packet image carefully and extract specific information for a garden planting database.

        First, examine the image closely and identify the type of seed packet shown.

        Then, extract the following information in a structured JSON format:
        - name: The main plant type (e.g., "Tomato", "Basil", "Carrot")
        - variety: The specific variety name (e.g., "Cherry Sweet", "Genovese", "Nantes")
        - brand: The company/manufacturer of the seed packet
        - germination_rate: The germination rate as a decimal (convert from percentage if needed, e.g., 85% → 0.85)
        - maturity: Days to maturity/harvest as an integer number only
        - growth: Growth habit (e.g., "Determinate", "Bush", "Vining", "Upright")
        - seed_depth: Recommended planting depth in inches (convert from other units if needed)
        - spacing: Recommended spacing between plants in inches (convert from other units if needed)
        - quantity: Number of seeds in the packet if mentioned
        - notes: Any special growing instructions, sun/water requirements, or other important information

        Your response should contain ONLY a valid JSON object with these fields. 
        Use null for any fields not found in the image.
        Format your response as a proper JSON object.
        """

        # Process the image to ensure it's under size limit
        try:
            image_bytes = self.image_processor.resize_image_for_gemini(image_path)
            img = Image.open(BytesIO(image_bytes))
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}

        try:
            # For Gemini with google-generativeai package
            generation_config = genai.types.GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json",  # Request JSON response if supported
            )

            response = await asyncio.to_thread(
                self.model.generate_content,
                [prompt, img],
                generation_config=generation_config,
            )

            if hasattr(response, "text"):
                try:
                    # Try to find and parse JSON in the response
                    content_text = response.text
                    json_str = content_text

                    # Multiple ways to extract JSON
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1].split("```")[0].strip()

                    # If we still don't have valid JSON, try to extract by brackets
                    if not (json_str.startswith("{") and json_str.endswith("}")):
                        if "{" in json_str and "}" in json_str:
                            start = json_str.find("{")
                            end = json_str.rfind("}") + 1
                            json_str = json_str[start:end]

                    # Additional cleanup of common issues
                    json_str = json_str.replace('\\"', '"')  # Fix escaped quotes
                    json_str = json_str.replace("\\n", " ")  # Fix newlines

                    structured_data = json.loads(json_str)

                    # Validate required fields
                    required_fields = [
                        "name",
                        "variety",
                        "brand",
                        "seed_depth",
                        "spacing",
                        "notes",
                    ]
                    missing_fields = [
                        field
                        for field in required_fields
                        if field not in structured_data
                    ]

                    if missing_fields:
                        print(
                            f"Warning: Missing required fields: {', '.join(missing_fields)}"
                        )

                    return structured_data
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


async def process_image(
    image_path: str, model_name: str = "gemini-2.5-pro-preview-03-25", genai_client=None
) -> tuple:
    """
    Process an image with Gemini Vision API and extract both OCR text and structured data.
    This function is designed to be called from the main app.

    Args:
        image_path: Path to the image file
        model_name: Name of the Gemini model to use
        genai_client: Optional preconfigured genai client

    Returns:
        Tuple of (ocr_text, structured_data)
    """
    # Initialize the Gemini Vision tester with the specified model
    vision_tester = GeminiVisionTester(model_name)

    # Extract OCR text
    ocr_result = await vision_tester.extract_ocr_text(str(image_path))
    ocr_text = ocr_result.get("text", "Error extracting OCR text")

    # Extract structured data
    structured_data = await vision_tester.extract_structured_data(str(image_path))

    # Return the results as a tuple
    return ocr_text, structured_data


async def main():
    """
    Run a test of the Gemini Vision API on seed packet images in the uploads directory.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test Gemini Vision API with seed packet images"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro-preview-03-25",
        help="Gemini model name to use (default: gemini-2.5-pro-preview-03-25)",
    )
    args = parser.parse_args()

    model_name = args.model
    print(f"Using Gemini model: {model_name}")

    # Find available images in the uploads directory
    uploads_dir = Path(__file__).parent.parent / "uploads"
    if not uploads_dir.exists():
        print(f"Uploads directory not found: {uploads_dir}")
        return

    # List image files
    image_files = []
    for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
        image_files.extend([f for f in uploads_dir.glob(ext) if f.is_file()])

    if not image_files:
        print("No images found in the uploads directory.")
        return

    print(f"Found {len(image_files)} image(s) to process.")

    # Initialize the Gemini Vision tester with the specified model
    vision_tester = GeminiVisionTester(model_name)

    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"\n{'='*60}")
        print(f"Processing image {i+1}/{len(image_files)}: {image_path.name}")
        print(f"{'='*60}")

        # Extract OCR text
        print("\n1. Extracting raw OCR text using Gemini Vision API...")
        try:
            ocr_result = await vision_tester.extract_ocr_text(str(image_path))

            if "error" not in ocr_result:
                print("\nExtracted OCR text:")
                print("-" * 50)
                extracted_text = ocr_result["text"]
                # Print first 500 chars with ellipsis if longer
                print(
                    extracted_text[:500] + ("..." if len(extracted_text) > 500 else "")
                )
                print("-" * 50)
            else:
                print(f"\nError extracting OCR text: {ocr_result['error']}")
        except Exception as e:
            print(f"\nException during OCR extraction: {str(e)}")

        # Extract structured data
        print("\n2. Extracting structured data using Gemini Vision API...")
        try:
            structured_data = await vision_tester.extract_structured_data(
                str(image_path)
            )

            if "error" not in structured_data:
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
                    if field not in structured_data or structured_data[field] is None
                ]
                if missing_fields:
                    print("\nMissing important fields:", ", ".join(missing_fields))
                else:
                    print("\nAll key fields were extracted successfully.")
            else:
                print(f"\nError extracting structured data: {structured_data['error']}")
                if "raw_response" in structured_data:
                    print("\nRaw response from Gemini:")
                    print(
                        structured_data["raw_response"][:500] + "..."
                        if len(structured_data["raw_response"]) > 500
                        else structured_data["raw_response"]
                    )
        except Exception as e:
            print(f"\nException during structured data extraction: {str(e)}")

        # Wait between images to avoid rate limits
        if i < len(image_files) - 1:
            print("\nWaiting 2 seconds before processing next image...")
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
