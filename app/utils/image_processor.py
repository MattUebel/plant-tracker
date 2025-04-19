"""
Image processing utility for the plant tracker application.
Handles image uploads, OCR processing, and structured data extraction using Anthropic's Claude API.
"""

import os
import uuid
import shutil
import base64
import json
import logging
import asyncio
from fastapi import UploadFile
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from io import BytesIO
from anthropic import Anthropic
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("image_processor")


class ImageProcessor:
    """Utility class for handling image uploads and OCR processing using Anthropic's Claude API"""

    def __init__(self):
        self.upload_dir = os.path.join(os.getcwd(), "uploads")
        # Get vision API configuration from environment
        self.vision_api_provider = os.getenv("VISION_API_PROVIDER", "claude").lower()

        # Claude API configuration
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.claude_model = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-20241022")

        # Gemini API configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv(
            "GEMINI_MODEL", "gemini-2.5-pro-preview-03-25"
        )  # Updated default

        self.mistral_model = os.getenv(
            "MISTRAL_MODEL", "pixtral-large-latest"
        )  # Use pixtral as default

        # Mistral API Key
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")

        self.max_retries = 3
        self.initial_retry_delay = 1.0
        self.max_image_size_bytes = 4.5 * 1024 * 1024  # 4.5MB max size for uploads
        self.min_quality = 60  # Don't go below 60% quality to maintain readability
        self.quality_step = 5  # Reduce quality in smaller steps
        self.scale_step = 0.95  # Scale dimensions by 5% at a time
        self.claude_max_size = 5 * 1024 * 1024  # Claude's 5MB limit

        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)

        # Initialize API clients based on provider configuration
        self.client = None
        self.gemini_client = None
        self.mistral_client = None  # Add mistral client placeholder

        # Log configuration status for the selected provider
        logger.info(f"Vision API provider configured: {self.vision_api_provider}")

        # Initialize the appropriate API client based on the configured provider
        if self.vision_api_provider == "claude":
            if self.anthropic_api_key:
                try:
                    self.client = Anthropic(api_key=self.anthropic_api_key)
                    logger.info(
                        f"ANTHROPIC_API_KEY configured successfully, using model: {self.claude_model}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize Anthropic client: {str(e)}")
            else:
                logger.warning(
                    "ANTHROPIC_API_KEY not found in environment variables. Claude image processing will not work."
                )
        elif self.vision_api_provider == "gemini":
            if self.gemini_api_key:
                # Import Gemini libraries only if needed
                try:
                    import google.generativeai as genai

                    genai.configure(api_key=self.gemini_api_key)
                    self.gemini_client = genai
                    logger.info(
                        f"GEMINI_API_KEY configured successfully, using model: {self.gemini_model}"
                    )
                except ImportError:
                    logger.warning(
                        "Failed to import google.generativeai. Please install it with: pip install google-generativeai"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini client: {str(e)}")
            else:
                logger.warning(
                    "GEMINI_API_KEY not found in environment variables. Gemini image processing will not work."
                )
        elif self.vision_api_provider == "mistral":
            if self.mistral_api_key:
                try:
                    from mistralai.client import MistralClient
                    from mistralai.models.chat_completion import ChatMessage

                    self.mistral_client = MistralClient(api_key=self.mistral_api_key)
                    logger.info(
                        f"MISTRAL_API_KEY configured successfully, using model: {self.mistral_model}"
                    )
                except ImportError:
                    logger.warning(
                        "Failed to import mistralai. Please install it with: pip install mistralai"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize Mistral client: {str(e)}")
            else:
                logger.warning(
                    "MISTRAL_API_KEY not found in environment variables. Mistral image processing will not work."
                )
        else:
            logger.warning(
                f"Unknown vision API provider: {self.vision_api_provider}. Defaulting to none."
            )

    def delete_image_file(self, filename: str) -> bool:
        """Delete an image file from the uploads directory"""
        try:
            file_path = os.path.join(self.upload_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Successfully deleted image file: {filename}")
                return True
            else:
                logger.warning(f"Image file not found for deletion: {filename}")
                return False
        except Exception as e:
            logger.error(f"Error deleting image file {filename}: {str(e)}")
            return False

    def resize_image_maintaining_quality(
        self, img: Image.Image, max_size_bytes: int
    ) -> Tuple[Image.Image, int]:
        """
        Resize an image while trying to maintain maximum quality possible.
        Returns the processed image and the final quality used.
        """
        width, height = img.size
        quality = 95  # Start with high quality

        # First try just reducing quality in small steps
        while quality >= self.min_quality:
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="JPEG", quality=quality)
            if img_byte_arr.tell() <= max_size_bytes:
                return img, quality
            quality -= self.quality_step

        # If quality reduction alone isn't enough, start scaling down dimensions
        # but reset quality to a higher value first
        quality = 85
        while True:
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="JPEG", quality=quality)
            if img_byte_arr.tell() <= max_size_bytes:
                return img, quality

            # Scale down dimensions
            width = int(width * self.scale_step)
            height = int(height * self.scale_step)
            img = img.resize((width, height), Image.LANCZOS)

            # If dimensions get too small, try reducing quality again
            if width < 800 or height < 800:
                quality -= self.quality_step
                if quality < self.min_quality:
                    # If we hit minimum quality and size is still too big,
                    # continue scaling dimensions as last resort
                    quality = self.min_quality

    async def save_image(
        self, file: UploadFile, entity_type: str, entity_id: int
    ) -> Dict[str, Any]:
        """Save an uploaded image to the uploads directory"""
        try:
            # Generate a unique filename using UUID as the primary identifier
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_id = str(uuid.uuid4().hex)
            ext = (
                os.path.splitext(file.filename)[1].lower() if file.filename else ".jpg"
            )
            filename = f"img_{unique_id}_{timestamp}{ext}"
            file_path = os.path.join(self.upload_dir, filename)

            # Read the uploaded file into memory
            contents = await file.read()
            img = Image.open(BytesIO(contents))

            # Convert to RGB if needed (removes alpha channel)
            if img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            ):
                img = img.convert("RGB")

            # Get original dimensions for logging
            original_width, original_height = img.size
            original_size = len(contents)

            # Process the image
            processed_img, final_quality = self.resize_image_maintaining_quality(
                img, self.max_image_size_bytes
            )

            # Save the processed image
            processed_img.save(file_path, "JPEG", quality=final_quality)
            final_size = os.path.getsize(file_path)
            final_width, final_height = processed_img.size

            # Log the transformation details
            logger.info(
                f"Image processed: {filename}\n"
                f"Original: {original_width}x{original_height}, {original_size/1024/1024:.1f}MB\n"
                f"Final: {final_width}x{final_height}, {final_size/1024/1024:.1f}MB, quality={final_quality}%"
            )

            return {
                "filename": filename,
                "file_path": file_path,
                "original_filename": file.filename,
                "mime_type": "image/jpeg",
                "file_size": final_size,
                "entity_type": entity_type,
                "entity_id": entity_id,
            }
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            raise
        finally:
            await file.close()

    def resize_image_for_claude(
        self, image_path: str, max_size_bytes: int = 4.5 * 1024 * 1024
    ) -> bytes:
        """
        Resize and compress an image to ensure it's under Claude's size limit.

        Args:
            image_path: Path to the image file
            max_size_bytes: Maximum size in bytes (default: 4.5MB to have safety margin)

        Returns:
            Processed image as bytes
        """
        try:
            # First check file size directly
            file_size = os.path.getsize(image_path)
            logger.info(f"Original image size: {file_size / (1024 * 1024):.2f} MB")

            # If already under size limit, can avoid processing
            if file_size <= max_size_bytes:
                logger.info("Image already under size limit, using as-is")
                with open(image_path, "rb") as f:
                    return f.read()

            # Need to resize/compress
            with Image.open(image_path) as img:
                # Convert to RGB if needed (removes alpha channel)
                if img.mode in ("RGBA", "LA") or (
                    img.mode == "P" and "transparency" in img.info
                ):
                    img = img.convert("RGB")

                original_width, original_height = img.size

                # Process the image with progressively more aggressive resizing
                processed_img, final_quality = self.resize_image_maintaining_quality(
                    img, max_size_bytes
                )

                # Convert to bytes
                img_byte_arr = BytesIO()
                processed_img.save(img_byte_arr, format="JPEG", quality=final_quality)
                img_byte_arr.seek(0)
                final_bytes = img_byte_arr.getvalue()

                logger.info(
                    f"Image resized for API: {original_width}x{original_height} -> "
                    f"{processed_img.width}x{processed_img.height}, "
                    f"Size: {len(final_bytes) / (1024 * 1024):.2f} MB, Quality: {final_quality}%"
                )

                return final_bytes

        except Exception as e:
            logger.error(f"Error resizing image for Claude: {str(e)}")
            # Fall back to reading the original file if something goes wrong
            with open(image_path, "rb") as f:
                return f.read()

    async def call_claude_api_with_retry(
        self, image_path: str, prompt: str, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call Claude API with an image and prompt using exponential backoff retry logic.
        Also handles progressive image resizing if size errors occur.

        Args:
            image_path: Path to the image file
            prompt: The prompt to send to Claude
            model_name: Optional specific model name to use

        Returns:
            Claude API response
        """
        if not self.client:
            error_msg = "Anthropic API key not configured. Set ANTHROPIC_API_KEY in your environment."
            logger.error(error_msg)
            return {"error": error_msg}

        retries = 0
        delay = self.initial_retry_delay
        last_exception = None
        current_max_size = (
            self.claude_max_size * 0.9
        )  # Start with 90% of max allowed size (safety margin)
        effective_model = model_name or self.claude_model

        while retries <= self.max_retries:
            try:
                # Process the image with current size target
                logger.info(
                    f"Processing image (attempt {retries+1}) with target size: {current_max_size/(1024*1024):.2f}MB"
                )
                image_bytes = self.resize_image_for_claude(
                    image_path, max_size_bytes=current_max_size
                )
                # Ensure base64-encoded payload stays under Claude's 5MB limit
                encoded_bytes = base64.b64encode(image_bytes)
                if len(encoded_bytes) > self.claude_max_size:
                    logger.warning(
                        f"Encoded image too large for Claude (base64): {len(encoded_bytes)/(1024*1024):.2f}MB. Retrying with more aggressive reduction."
                    )
                    current_max_size *= 0.8  # Reduce target size by 20%
                    retries += 1
                    continue
                encoded_image = encoded_bytes.decode()

                logger.info(
                    f"Image processed. Final size: {len(encoded_bytes)/(1024*1024):.2f} MB"
                )

                # Call the API with the processed image
                logger.info(
                    f"Calling Claude API (model: {effective_model}, attempt {retries+1}/{self.max_retries+1})..."
                )
                response = self.client.messages.create(
                    model=effective_model,
                    max_tokens=4000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
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
                logger.info("Successfully received response from Claude API")
                return response

            except Exception as e:
                error_str = str(e)
                logger.error(f"API call failed: {error_str}")

                # If we get a specific size error, immediately try with more aggressive resizing
                if (
                    "image exceeds" in error_str.lower()
                    or "maximum allowed size" in error_str.lower()
                ):
                    current_max_size *= 0.7  # Reduce target size by 30%
                    logger.info(
                        f"Size error detected. Reducing target size to {current_max_size/(1024*1024):.2f}MB"
                    )

                if retries < self.max_retries:
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                    retries += 1
                    last_exception = e
                else:
                    error_msg = f"Max retries exceeded: {str(last_exception)}"
                    logger.error(error_msg)
                    return {"error": error_msg}

        return {"error": "Max retries exceeded with unknown error"}

    async def call_mistral_chat_api_with_retry(
        self, image_path: str, prompt: str, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call Mistral Chat API with an image and prompt using exponential backoff retry logic.
        Uses a single call with a multimodal model (like pixtral).

        Args:
            image_path: Path to the image file
            prompt: The prompt to send (including schema instructions)
            model_name: Optional specific model name to use

        Returns:
            Mistral API response content as a dictionary or an error dict
        """
        if not self.mistral_client:
            error_msg = "Mistral API key/client not configured. Set MISTRAL_API_KEY and install mistralai."
            logger.error(error_msg)
            return {"error": error_msg}

        retries = 0
        delay = self.initial_retry_delay
        last_exception = None
        effective_model = model_name or self.mistral_model

        try:
            # Encode the image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            # Determine mime type (simple approach)
            mime_type = (
                "image/jpeg"
                if image_path.lower().endswith((".jpg", ".jpeg"))
                else "image/png"
            )

        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            return {"error": f"Failed to encode image: {str(e)}"}

        # Import ChatMessage here to avoid import errors if mistralai isn't installed
        try:
            from mistralai.models.chat_completion import ChatMessage
        except ImportError:
            logger.error(
                "MistralAI client library not found. Please install with: pip install mistralai"
            )
            return {"error": "MistralAI library not installed"}

        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{base64_image}",
                    },
                ],
            )
        ]

        while retries <= self.max_retries:
            try:
                logger.info(
                    f"Calling Mistral Chat API (model: {effective_model}, attempt {retries+1}/{self.max_retries+1})..."
                )
                start_time = time.perf_counter()
                chat_response = self.mistral_client.chat(
                    model=effective_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                end_time = time.perf_counter()
                logger.info(
                    f"Successfully received response from Mistral Chat API ({(end_time - start_time):.2f}s)"
                )

                if chat_response.choices and chat_response.choices[0].message.content:
                    try:
                        # The response content should be a JSON string
                        structured_data = json.loads(
                            chat_response.choices[0].message.content
                        )
                        return structured_data
                    except json.JSONDecodeError as json_err:
                        logger.error(f"Mistral response was not valid JSON: {json_err}")
                        logger.debug(
                            f"Raw Mistral response content: {chat_response.choices[0].message.content}"
                        )
                        return {"error": "Failed to parse JSON response from Mistral"}
                else:
                    logger.warning("Mistral API returned no content in choices.")
                    return {"error": "Mistral API returned no content"}

            except Exception as e:
                # Check if it's a Mistral specific error if possible, otherwise generic handling
                error_str = str(e)
                logger.error(f"Mistral API call failed: {error_str}")
                last_exception = e

                if retries < self.max_retries:
                    logger.info(f"Retrying Mistral call in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                    retries += 1
                else:
                    error_msg = (
                        f"Max retries exceeded for Mistral API: {str(last_exception)}"
                    )
                    logger.error(error_msg)
                    return {"error": error_msg}

        return {"error": "Max retries exceeded for Mistral API with unknown error"}

    async def process_image_with_vision_api(
        self,
        image_path: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Process image with the configured Vision API and extract text and structured data"""
        effective_provider = (provider or self.vision_api_provider).lower()
        effective_model_name = model_name  # Keep explicit model if passed
        log_model = effective_model_name or "default"

        logger.info(
            f"Processing image with {effective_provider.upper()} Vision API (Model: {log_model}) for: {image_path}"
        )

        if effective_provider == "claude":
            if not self.anthropic_api_key or not self.client:
                error_msg = "Claude API key not configured. Set ANTHROPIC_API_KEY in your environment."
                logger.error(error_msg)
                return error_msg, None
            return await self.process_ocr(image_path, model_name=effective_model_name)

        elif effective_provider == "gemini":
            if not self.gemini_api_key or not self.gemini_client:
                error_msg = "Gemini API key not configured. Set GEMINI_API_KEY in your environment."
                logger.error(error_msg)
                return error_msg, None

            try:
                # Import the Gemini module
                import sys
                import os

                # Add the current directory to the path to ensure modules can be found
                sys.path.insert(0, os.getcwd())

                effective_model = effective_model_name or self.gemini_model
                try:
                    # Import the Gemini module directly
                    from utils.gemini_vision_api import GeminiVisionTester

                    # Initialize the vision tester with the effective model
                    vision_tester = GeminiVisionTester(effective_model)

                    # First get OCR text
                    logger.info(
                        f"Extracting OCR text with Gemini (model: {effective_model}) from: {image_path}"
                    )
                    ocr_result = await vision_tester.extract_ocr_text(image_path)
                    ocr_text = ocr_result.get("text", "Error extracting OCR text")

                    # Try extracting structured data with enhanced method
                    logger.info(
                        f"Extracting structured data with Gemini (model: {effective_model}) from: {image_path}"
                    )
                    structured_data = await vision_tester.extract_structured_data(
                        image_path
                    )

                    # If we got an error, try the JSON mode method as fallback
                    if "error" in structured_data:
                        logger.info("First method failed, trying JSON mode extraction")
                        structured_data = (
                            await vision_tester.extract_structured_data_with_json_mode(
                                image_path
                            )
                        )

                    # Remove error field if it exists
                    if isinstance(structured_data, dict) and "error" in structured_data:
                        logger.warning(
                            f"Gemini structured data extraction error: {structured_data['error']}"
                        )
                        structured_data = None

                    logger.info("Successfully processed image with Gemini Vision API")
                    return ocr_text, structured_data

                except ImportError:
                    # Fall back to the simpler process_image function if module structure differs
                    logger.warning(
                        "Couldn't import GeminiVisionTester, falling back to process_image"
                    )
                    from utils.gemini_vision_api import process_image

                    # Pass the effective model to the fallback function
                    return await process_image(
                        image_path, effective_model, self.gemini_client
                    )

            except Exception as e:
                error_msg = f"Error processing with Gemini Vision API: {str(e)}"
                logger.error(error_msg)
                return error_msg, None

        elif effective_provider == "mistral":
            if not self.mistral_api_key or not self.mistral_client:
                error_msg = "Mistral API key/client not configured. Set MISTRAL_API_KEY and install mistralai."
                logger.error(error_msg)
                return error_msg, None

            try:
                # Define the structured data prompt for Mistral (similar to Claude/Gemini)
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

                # Call the single-step Mistral chat API
                structured_data_result = await self.call_mistral_chat_api_with_retry(
                    image_path, structured_prompt, model_name=effective_model_name
                )

                if "error" in structured_data_result:
                    logger.error(
                        f"Mistral processing failed: {structured_data_result['error']}"
                    )
                    # Return empty OCR text and None for structured data on failure
                    return f"Mistral Error: {structured_data_result['error']}", None
                else:
                    logger.info(
                        "Successfully processed image with Mistral Vision API (single step)"
                    )
                    # Return empty string for OCR text (as it's not extracted separately) and the structured data
                    return "", structured_data_result

            except Exception as e:
                error_msg = f"Error processing with Mistral Vision API: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return error_msg, None

        else:
            error_msg = f"Unsupported vision API provider: {effective_provider}"
            logger.error(error_msg)
            return error_msg, None


# Initialize a singleton instance for app usage
image_processor = ImageProcessor()
