"""
Image processing utility for the plant tracker application.
Handles image uploads, OCR processing, and structured data extraction.
"""

import os
import uuid
import shutil
import base64
import httpx
import json
import cv2
import numpy as np
import logging
import asyncio  # Added missing import for asyncio
from fastapi import UploadFile
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("image_processor")

# Lazy imports for vision APIs
try:
    from anthropic import Anthropic

    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    logger.warning(
        "anthropic package not installed. Claude vision API will not be available."
    )

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning(
        "google-generativeai package not installed. Gemini vision API will not be available."
    )


class ImageProcessor:
    """Utility class for handling image uploads and OCR processing"""

    def __init__(self):
        self.upload_dir = os.path.join(os.getcwd(), "uploads")

        # API keys and configuration
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        self.mistral_chat_api_endpoint = "https://api.mistral.ai/v1/chat/completions"
        self.mistral_ocr_api_endpoint = "https://api.mistral.ai/v1/ocr"

        # Vision API configuration
        self.vision_api_provider = os.getenv("VISION_API_PROVIDER", "claude").lower()
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.claude_model = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")

        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)

        # Log configuration status
        if not self.mistral_api_key:
            logger.warning(
                "MISTRAL_API_KEY not found in environment variables. OCR functionality will not work."
            )
        else:
            logger.info("MISTRAL_API_KEY configured successfully")

        # Initialize Vision API clients if keys are available
        if self.vision_api_provider == "claude":
            if not self.claude_api_key:
                logger.warning(
                    "ANTHROPIC_API_KEY not found. Claude Vision API will not work."
                )
            elif not CLAUDE_AVAILABLE:
                logger.warning(
                    "anthropic package not installed. Claude Vision API will not work."
                )
            else:
                try:
                    self.claude_client = Anthropic(api_key=self.claude_api_key)
                    logger.info(
                        f"Claude Vision API configured with model: {self.claude_model}"
                    )
                except Exception as e:
                    logger.error(f"Error initializing Claude client: {str(e)}")
        elif self.vision_api_provider == "gemini":
            if not self.gemini_api_key:
                logger.warning(
                    "GEMINI_API_KEY not found. Gemini Vision API will not work."
                )
            elif not GEMINI_AVAILABLE:
                logger.warning(
                    "google-genai package not installed. Gemini Vision API will not work."
                )
            else:
                try:
                    genai.configure(api_key=self.gemini_api_key)
                    self.gemini_model_instance = genai.GenerativeModel(
                        self.gemini_model
                    )
                    logger.info(
                        f"Gemini Vision API configured with model: {self.gemini_model}"
                    )
                except Exception as e:
                    logger.error(f"Error initializing Gemini client: {str(e)}")
        else:
            logger.warning(
                f"Unknown vision API provider: {self.vision_api_provider}. Using Mistral for fallback."
            )

    async def save_image(
        self, file: UploadFile, entity_type: str, entity_id: int
    ) -> Dict[str, Any]:
        """Save an uploaded image to the uploads directory"""
        try:
            # Generate a unique filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_id = str(uuid.uuid4().hex[:8])
            ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
            filename = f"{entity_type}_{entity_id}_{timestamp}_{unique_id}{ext}"

            # Full path where the file will be saved
            file_path = os.path.join(self.upload_dir, filename)

            # Save the file
            with open(file_path, "wb") as image_file:
                shutil.copyfileobj(file.file, image_file)

            # Get file size
            file_size = os.path.getsize(file_path)

            logger.info(f"Successfully saved image: {filename} ({file_size} bytes)")

            return {
                "filename": filename,
                "file_path": file_path,
                "original_filename": file.filename,
                "mime_type": file.content_type,
                "file_size": file_size,
                "entity_type": entity_type,
                "entity_id": entity_id,
            }
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            raise
        finally:
            await file.close()

    def encode_image(self, image_path: str) -> str:
        """Encode an image to base64 for API transmission"""
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
                return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            raise

    def delete_image_file(self, filename: str) -> bool:
        """
        Delete an image file from the uploads directory.

        Args:
            filename: The filename of the image to delete (without path)

        Returns:
            bool: True if file was successfully deleted, False otherwise
        """
        try:
            file_path = os.path.join(self.upload_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Successfully deleted image: {filename}")
                return True
            else:
                logger.warning(f"Image file not found for deletion: {filename}")
                return False
        except Exception as e:
            logger.error(f"Error deleting image file {filename}: {str(e)}")
            return False

    async def process_ocr(
        self, image_path: str
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Process image with Mistral OCR API and extract text"""
        if not self.mistral_api_key:
            error_msg = "No Mistral API key found in environment variables. Set MISTRAL_API_KEY in your environment."
            logger.error(error_msg)
            return error_msg, None

        try:
            # Encode the original image - skip preprocessing as it reduces OCR quality
            logger.info(f"Encoding image for OCR processing: {image_path}")
            image_b64 = self.encode_image(image_path)

            # Prepare headers for the API request
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json",
            }

            # Prepare the request payload for OCR using the dedicated OCR API
            payload = {
                "model": "mistral-ocr-latest",
                "document": {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_b64}",
                },
            }

            # Make the API request to get OCR text
            logger.info("Sending request to Mistral OCR API")
            async with httpx.AsyncClient() as client:
                logger.debug(
                    f"Making request to endpoint: {self.mistral_ocr_api_endpoint}"
                )
                try:
                    response = await client.post(
                        self.mistral_ocr_api_endpoint,
                        headers=headers,
                        json=payload,
                        timeout=30.0,
                    )

                    logger.info(
                        f"Received response from Mistral OCR API: status_code={response.status_code}"
                    )

                    if response.status_code != 200:
                        error_msg = (
                            f"OCR API error: {response.status_code} - {response.text}"
                        )
                        logger.error(error_msg)
                        return error_msg, None

                    ocr_result = response.json()

                    # Extract text from the OCR response pages
                    ocr_text = ""
                    if "pages" in ocr_result:
                        for page in ocr_result["pages"]:
                            ocr_text += page.get("markdown", "")
                        logger.info(
                            f"Successfully extracted OCR text ({len(ocr_text)} characters)"
                        )
                    else:
                        error_msg = "OCR API response missing 'pages' field"
                        logger.error(error_msg)
                        logger.debug(f"Response content: {ocr_result}")
                        return error_msg, None

                except httpx.RequestError as e:
                    error_msg = f"Error making request to Mistral OCR API: {str(e)}"
                    logger.error(error_msg)
                    return error_msg, None

            # Now extract structured data using the Chat API
            structured_data = await self.extract_structured_data(ocr_text)

            return ocr_text, structured_data
        except Exception as e:
            error_msg = f"Error processing OCR: {str(e)}"
            logger.error(error_msg)
            return error_msg, None

    async def extract_structured_data(self, ocr_text: str) -> Optional[Dict[str, Any]]:
        """Extract structured data from OCR text using Mistral's language model"""
        if not ocr_text or ocr_text.strip() == "":
            logger.warning("Cannot extract structured data: OCR text is empty")
            return None

        try:
            # Prepare the prompt for structured data extraction
            logger.info("Preparing to extract structured data from OCR text")
            prompt = f"""
            Based on the following OCR text from a seed packet, extract structured data in a JSON format. Please parse the following fields:
            
            1. name (the basic type of seed, like "Tomato")
            2. variety (the specific variety, like "Cherokee Purple")
            3. germination_rate (as a decimal if found, like 0.85 for 85%)
            4. maturity (days to maturity as an integer)
            5. growth (growth habit, like "Determinate" or "Vining")
            6. seed_depth (planting depth in inches as a decimal)
            7. spacing (spacing between plants in inches as a decimal)
            8. notes (any additional important information)
            
            Only extract fields that are clearly present in the text, and use null for missing fields. Return ONLY a valid JSON object with these fields, nothing else.
            OCR TEXT:
            {ocr_text}
            """

            # Prepare the request
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": "mistral-small-latest",  # Can use small model for structured extraction
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }

            # Make the API request
            logger.info(
                "Sending request to Mistral Chat API for structured data extraction"
            )
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        self.mistral_chat_api_endpoint,
                        headers=headers,
                        json=payload,
                        timeout=30.0,
                    )

                    logger.info(
                        f"Received response from Mistral Chat API: status_code={response.status_code}"
                    )

                    if response.status_code != 200:
                        logger.error(
                            f"LLM API error: {response.status_code} - {response.text}"
                        )
                        return None

                    result = response.json()
                    structured_text = result["choices"][0]["message"]["content"]

                    # Parse JSON response
                    try:
                        structured_data = json.loads(structured_text)
                        logger.info("Successfully extracted structured data")
                        logger.debug(f"Structured data: {structured_data}")
                        return structured_data
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Error decoding JSON from LLM response: {structured_text}"
                        )
                        logger.error(f"JSON decode error: {str(e)}")
                        return None

                except httpx.RequestError as e:
                    logger.error(f"Error making request to Mistral Chat API: {str(e)}")
                    return None

        except Exception as e:
            logger.error(f"Error extracting structured data: {str(e)}")
            return None

    async def process_image_with_vision_api(
        self, image_path: str
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Process image with the configured vision API provider (Claude or Gemini)"""
        logger.info(
            f"Processing image with {self.vision_api_provider.capitalize()} Vision API: {image_path}"
        )

        if self.vision_api_provider == "claude":
            return await self._process_with_claude(image_path)
        elif self.vision_api_provider == "gemini":
            return await self._process_with_gemini(image_path)
        else:
            # Fallback to Mistral OCR
            logger.warning(
                f"No valid vision API provider configured. Falling back to Mistral OCR."
            )
            return await self.process_ocr(image_path)

    async def _process_with_claude(
        self, image_path: str
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Process image using Claude Vision API"""
        if not self.claude_api_key or not CLAUDE_AVAILABLE:
            error_msg = "Claude API key not set or anthropic package not installed"
            logger.error(error_msg)
            return error_msg, None

        try:
            # Process the image to fit within Claude's size limits
            logger.info(f"Processing image for Claude API: {image_path}")

            # Image preparation
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()

            # Resize image if needed (Claude has a 4MB limit)
            from io import BytesIO
            from PIL import Image

            img = Image.open(BytesIO(image_bytes))

            # Convert to RGB if needed
            if img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            ):
                img = img.convert("RGB")

            # Compress and resize if needed
            max_size_bytes = 4 * 1024 * 1024  # 4MB limit
            quality = 90
            width, height = img.size

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

            # Get the processed image as bytes and encode for API
            img_byte_arr.seek(0)
            image_bytes = img_byte_arr.getvalue()
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")

            # Extract OCR text first
            ocr_prompt = """
            Perform OCR on this seed packet image. 
            Extract ALL text visible in the image, preserving the layout as much as possible.
            Include all product information, instructions, and details exactly as they appear.
            Focus only on the text content - do not analyze or interpret the information.
            """

            # Make the API call for OCR
            logger.info("Sending OCR request to Claude API")
            try:
                ocr_response = self.claude_client.messages.create(
                    model=self.claude_model,
                    max_tokens=4000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": ocr_prompt},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": encoded_image,
                                    },
                                },
                            ],
                        }
                    ],
                )

                if hasattr(ocr_response, "content") and len(ocr_response.content) > 0:
                    ocr_text = ocr_response.content[0].text
                    logger.info(
                        f"Successfully extracted OCR text with Claude ({len(ocr_text)} characters)"
                    )
                else:
                    error_msg = "Failed to extract OCR text: Invalid response format"
                    logger.error(error_msg)
                    return error_msg, None

            except Exception as e:
                error_msg = f"Error calling Claude API for OCR: {str(e)}"
                logger.error(error_msg)
                return error_msg, None

            # Now extract structured data
            structured_prompt = """
            Extract structured plant information from this seed packet image.
            Return ONLY a valid JSON object matching the following fields and types:
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
            Your response should be ONLY the JSON object with no additional text.
            """

            # Make the API call for structured data
            logger.info("Sending structured data extraction request to Claude API")
            try:
                structured_response = self.claude_client.messages.create(
                    model=self.claude_model,
                    max_tokens=4000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": structured_prompt},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": encoded_image,
                                    },
                                },
                            ],
                        }
                    ],
                )

                if (
                    hasattr(structured_response, "content")
                    and len(structured_response.content) > 0
                ):
                    # Extract JSON from response
                    content_text = structured_response.content[0].text

                    # Find JSON object in the response text
                    json_str = content_text
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1].split("```")[0].strip()

                    try:
                        structured_data = json.loads(json_str)
                        logger.info(
                            "Successfully extracted structured data with Claude API"
                        )
                        logger.debug(f"Structured data: {structured_data}")
                        return ocr_text, structured_data
                    except json.JSONDecodeError as e:
                        error_msg = (
                            f"Failed to parse JSON response from Claude: {str(e)}"
                        )
                        logger.error(error_msg)
                        return ocr_text, None
                else:
                    error_msg = (
                        "Failed to extract structured data: Invalid response format"
                    )
                    logger.error(error_msg)
                    return ocr_text, None

            except Exception as e:
                error_msg = f"Error calling Claude API for structured data: {str(e)}"
                logger.error(error_msg)
                return ocr_text, None

        except Exception as e:
            error_msg = f"Error processing image with Claude API: {str(e)}"
            logger.error(error_msg)
            return error_msg, None

    async def _process_with_gemini(
        self, image_path: str
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Process image using Gemini Vision API following best practices from documentation"""
        if not self.gemini_api_key or not GEMINI_AVAILABLE:
            error_msg = (
                "Gemini API key not set or google-generativeai package not installed"
            )
            logger.error(error_msg)
            return error_msg, None

        try:
            # Process the image to fit within Gemini's size limits
            logger.info(f"Processing image for Gemini API: {image_path}")

            # Image preparation following best practices
            from io import BytesIO
            from PIL import Image

            try:
                with Image.open(image_path) as img:
                    # Convert to RGB if needed (recommended in docs)
                    if img.mode in ("RGBA", "LA") or (
                        img.mode == "P" and "transparency" in img.info
                    ):
                        img = img.convert("RGB")

                    # Start with original dimensions
                    width, height = img.size
                    quality = 90  # Recommended starting point in docs

                    # Create a BytesIO object to check size without saving to disk
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format="JPEG", quality=quality)
                    current_size = img_byte_arr.tell()

                    # Iteratively reduce size until under limit (4.5MB to be safe)
                    max_size_bytes = 4.5 * 1024 * 1024
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

                    # Keep a copy of the processed image
                    img_byte_arr.seek(0)
                    processed_img = Image.open(img_byte_arr)

                    logger.info(
                        f"Successfully processed image: size={current_size/1024/1024:.2f}MB, dimensions={width}x{height}"
                    )
            except Exception as img_error:
                logger.error(f"Error processing image file: {str(img_error)}")
                return f"Error processing image: {str(img_error)}", None

            # First - direct OCR extraction for text only
            ocr_prompt = """
            Perform OCR on this seed packet image. 
            Extract ALL text visible in the image, preserving the layout as much as possible.
            Include all product information, instructions, and details exactly as they appear.
            Focus only on the text content - do not analyze or interpret the information.
            """

            # Make OCR request first to get raw text
            logger.info("Sending OCR request to Gemini API")
            try:
                ocr_response = await asyncio.to_thread(
                    self.gemini_model_instance.generate_content,
                    [ocr_prompt, processed_img],
                    generation_config={"temperature": 0.1, "max_output_tokens": 2048},
                )

                if hasattr(ocr_response, "text"):
                    ocr_text = ocr_response.text
                    logger.info(
                        f"Successfully extracted OCR text: {len(ocr_text)} chars"
                    )
                else:
                    ocr_text = "Failed to extract OCR text from image"
                    logger.warning("OCR text extraction failed")
            except Exception as ocr_error:
                logger.error(f"Error in OCR extraction: {str(ocr_error)}")
                ocr_text = f"OCR extraction error: {str(ocr_error)}"

            # Now extract structured data with a clear JSON response format
            structured_prompt = """
            Analyze this seed packet image and extract the following information as a JSON object:
            - name: The main plant type (e.g., 'Tomato', 'Basil', 'Carrot')
            - variety: The specific variety name (e.g., 'Cherry Sweet', 'Genovese', 'Nantes')
            - brand: The company/manufacturer of the seed packet
            - seed_depth: Recommended planting depth in inches (convert from other units if needed)
            - spacing: Recommended spacing between plants in inches (convert from other units if needed)
            - notes: Any special growing instructions or other important information

            Return ONLY a valid JSON object with these fields. Use null for any fields not found in the image.
            """

            # Make the API call for structured data
            logger.info("Sending structured data extraction request to Gemini API")
            try:
                structured_response = await asyncio.to_thread(
                    self.gemini_model_instance.generate_content,
                    [structured_prompt, processed_img],
                    generation_config={
                        "temperature": 0.1,
                        "top_p": 0.95,
                        "max_output_tokens": 1024,
                        "response_mime_type": "application/json",  # Request JSON format
                    },
                )

                logger.info(f"Received response from Gemini API")

                # Handle empty candidates - this is the specific error we're fixing
                if not hasattr(structured_response, "text"):
                    logger.warning("Empty response from Gemini API (no candidates)")
                    return ocr_text, self._create_fallback_data(
                        "Error during structured data extraction: The model returned an empty response. Please enter details manually."
                    )

                # Extract and clean the response text
                response_text = structured_response.text
                logger.info(f"Got response from Gemini ({len(response_text)} chars)")

                # Clean the text to extract just the JSON portion
                json_str = self._extract_json_from_text(response_text)

                if json_str:
                    try:
                        structured_data = json.loads(json_str)
                        logger.info("Successfully parsed structured data JSON")

                        # Validate and provide defaults for missing required fields
                        structured_data = self._validate_seed_data(structured_data)
                        return ocr_text, structured_data

                    except json.JSONDecodeError as json_err:
                        logger.error(f"Failed to parse JSON: {str(json_err)}")
                        logger.debug(f"Problematic JSON string: {json_str}")
                        # Return OCR text with fallback data
                        return ocr_text, self._create_fallback_data(
                            "Couldn't parse structured data from image. Please review OCR text and enter details manually."
                        )
                else:
                    logger.warning("No valid JSON found in Gemini response")
                    return ocr_text, self._create_fallback_data(
                        "No structured data found in the response. Please review OCR text and enter details manually."
                    )

            except Exception as api_error:
                # Specifically catch and handle the empty candidates error
                error_message = str(api_error)
                if "response.candidates` is empty" in error_message:
                    logger.warning("Gemini API returned empty candidates")
                    return ocr_text, self._create_fallback_data(
                        "Error during structured data extraction: Invalid operation with empty response candidates. Please enter details manually."
                    )

                logger.error(
                    f"Error calling Gemini API for structured data: {error_message}"
                )
                # Return OCR text with fallback data
                return ocr_text, self._create_fallback_data(
                    f"Error during structured data extraction: {error_message}. Please enter details manually."
                )

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            return f"Error processing image: {str(e)}", None

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON object from text that may contain markdown or other content"""
        if not text:
            return None

        # Case 1: The text is already valid JSON
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Case 2: JSON in markdown code block
        if "```json" in text:
            # Extract content between ```json and ``` markers
            parts = text.split("```json")
            if len(parts) > 1:
                json_part = parts[1].split("```")[0].strip()
                return json_part

        # Case 3: JSON in generic code block
        if "```" in text:
            # Extract content between ``` markers
            parts = text.split("```")
            if len(parts) > 1:
                json_part = parts[1].strip()
                # Verify it looks like JSON
                if json_part.startswith("{") and json_part.endswith("}"):
                    return json_part

        # Case 4: Just look for JSON object between curly braces
        if "{" in text and "}" in text:
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            if start_idx < end_idx:
                json_part = text[start_idx:end_idx]
                # Verify it's valid JSON
                try:
                    json.loads(json_part)
                    return json_part
                except json.JSONDecodeError:
                    pass

        # No valid JSON found
        return None

    def _validate_seed_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize seed data from API responses"""
        # Ensure all expected fields exist (with None if missing)
        validated = {
            "name": data.get("name"),
            "variety": data.get("variety"),
            "brand": data.get("brand"),
            "germination_rate": None,
            "maturity": None,
            "growth": data.get("growth"),
            "seed_depth": None,
            "spacing": None,
            "notes": data.get("notes"),
        }

        # Convert numeric fields properly
        if "germination_rate" in data and data["germination_rate"]:
            try:
                # Handle percentage strings like "85%" by converting to decimal
                if isinstance(data["germination_rate"], str):
                    germination = data["germination_rate"].strip()
                    if "%" in germination:
                        germination = germination.replace("%", "").strip()
                        validated["germination_rate"] = float(germination) / 100
                    else:
                        # If it's a decimal string like "0.85"
                        validated["germination_rate"] = float(germination)
                elif isinstance(data["germination_rate"], (int, float)):
                    # If it's already a number
                    if data["germination_rate"] > 1:
                        # Convert from percentage to decimal if > 1
                        validated["germination_rate"] = data["germination_rate"] / 100
                    else:
                        validated["germination_rate"] = data["germination_rate"]
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid germination_rate format: {data.get('germination_rate')}"
                )

        # Convert maturity days to integer
        if "maturity" in data and data["maturity"]:
            try:
                if isinstance(data["maturity"], str):
                    # Extract numbers from strings like "60-75 days"
                    import re

                    numbers = re.findall(r"\d+", data["maturity"])
                    if numbers:
                        # Use the first number or average if there's a range
                        if len(numbers) == 1:
                            validated["maturity"] = int(numbers[0])
                        else:
                            # Average if it's a range like "60-75 days"
                            validated["maturity"] = int(
                                sum(int(n) for n in numbers) / len(numbers)
                            )
                elif isinstance(data["maturity"], (int, float)):
                    validated["maturity"] = int(data["maturity"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid maturity format: {data.get('maturity')}")

        # Convert seed depth to float
        if "seed_depth" in data and data["seed_depth"]:
            try:
                if isinstance(data["seed_depth"], str):
                    # Handle strings like "1/4 inch" or "0.25 inches"
                    depth_str = data["seed_depth"].lower()
                    if "/" in depth_str:
                        # Handle fractions like "1/4 inch"
                        fraction_parts = depth_str.split()[0].split("/")
                        if len(fraction_parts) == 2:
                            validated["seed_depth"] = float(fraction_parts[0]) / float(
                                fraction_parts[1]
                            )
                    else:
                        # Extract first number
                        import re

                        numbers = re.findall(r"\d+\.?\d*", depth_str)
                        if numbers:
                            validated["seed_depth"] = float(numbers[0])
                elif isinstance(data["seed_depth"], (int, float)):
                    validated["seed_depth"] = float(data["seed_depth"])
            except (ValueError, TypeError, ZeroDivisionError):
                logger.warning(f"Invalid seed_depth format: {data.get('seed_depth')}")

        # Convert spacing to float
        if "spacing" in data and data["spacing"]:
            try:
                if isinstance(data["spacing"], str):
                    # Handle strings like "6-8 inches" or "0.25 feet"
                    spacing_str = data["spacing"].lower()
                    # Extract numbers
                    import re

                    numbers = re.findall(r"\d+\.?\d*", spacing_str)
                    if numbers:
                        # Use average if range
                        if len(numbers) > 1:
                            spacing = sum(float(n) for n in numbers) / len(numbers)
                        else:
                            spacing = float(numbers[0])

                        # Convert to inches if needed
                        if "feet" in spacing_str or "foot" in spacing_str:
                            spacing *= 12
                        validated["spacing"] = spacing
                elif isinstance(data["spacing"], (int, float)):
                    validated["spacing"] = float(data["spacing"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid spacing format: {data.get('spacing')}")

        return validated

    def _create_fallback_data(self, message: str) -> Dict[str, Any]:
        """Create a basic fallback data structure with a message in notes field"""
        return {
            "name": None,
            "variety": None,
            "brand": None,
            "germination_rate": None,
            "maturity": None,
            "growth": None,
            "seed_depth": None,
            "spacing": None,
            "notes": message,
        }


# Initialize a singleton instance for app usage
image_processor = ImageProcessor()
