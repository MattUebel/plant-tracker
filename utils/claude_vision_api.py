"""
Utility script to test Anthropic's Claude API with seed packet images.
This script extracts both raw OCR text and structured data from seed packet images
based on the application's Seed model schema.

Usage:
    cd /Users/mattuebel/mattuebel/plant-tracker
    python -m utils.claude_vision_api [model_name]
    
    Default model: claude-3-5-haiku-20241022
    Example with different model: python -m utils.claude_vision_api claude-3-5-sonnet-20241022
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
from anthropic import Anthropic
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Load environment variables from project root
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# TODO: For testing only - remove or replace with your actual API key before running
if not os.getenv("ANTHROPIC_API_KEY"):
    print("Warning: ANTHROPIC_API_KEY not found in environment variables.")
    print("Please set your ANTHROPIC_API_KEY using:")
    print("export ANTHROPIC_API_KEY='your_api_key_here'")
    print("Or add it to your .env file.")
    sys.exit(1)

class ImageProcessor:
    """Image processing utilities to prepare images for Claude API."""
    
    @staticmethod
    def resize_image_for_claude(image_path: str, max_size_bytes: int = 4.5 * 1024 * 1024) -> bytes:
        """
        Resize and compress an image to ensure it's under Claude's size limit.
        
        Args:
            image_path: Path to the image file
            max_size_bytes: Maximum size in bytes (default: 4.5MB to have safety margin)
            
        Returns:
            Processed image as bytes
        """
        # Open the image with PIL for better quality control
        with Image.open(image_path) as img:
            # Convert to RGB if needed (removes alpha channel)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGB')
            
            # Start with original dimensions
            width, height = img.size
            quality = 90
            
            # Create a BytesIO object to check size without saving to disk
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=quality)
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
                    img = img.resize((width, height), Image.LANCZOS)
                
                # Check new size
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=quality)
                current_size = img_byte_arr.tell()
            
            # Get the processed image as bytes
            img_byte_arr.seek(0)
            return img_byte_arr.getvalue()

class ClaudeAPICaller:
    """Base class for Anthropic Claude API calls with retry logic."""
    
    def __init__(self, model_name: str = "claude-3-5-haiku-20241022"):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = model_name
        self.max_retries = 3
        self.initial_retry_delay = 1.0
        self.image_processor = ImageProcessor()

    async def call_api_with_retry(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Call Claude API with an image and prompt using exponential backoff retry logic.
        
        Args:
            image_path: Path to the image file
            prompt: The prompt to send to Claude
            
        Returns:
            Claude API response
        """
        retries = 0
        delay = self.initial_retry_delay
        last_exception = None
        
        # Process the image to ensure it's under Claude's size limit
        try:
            print(f"Processing image to fit within size limits...")
            image_bytes = self.image_processor.resize_image_for_claude(image_path)
            encoded_image = base64.b64encode(image_bytes).decode()
            image_size_mb = len(image_bytes) / (1024 * 1024)
            print(f"Image processed. New size: {image_size_mb:.2f} MB")
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise
        
        while retries <= self.max_retries:
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text", 
                                    "text": prompt
                                },
                                {
                                    "type": "image", 
                                    "source": {
                                        "type": "base64", 
                                        "media_type": "image/jpeg", 
                                        "data": encoded_image
                                    }
                                }
                            ]
                        }
                    ]
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

class ClaudeVisionTester(ClaudeAPICaller):
    """Test Claude Vision API for image understanding and data extraction."""
    
    def __init__(self, model_name: str = "claude-3-5-haiku-20241022"):
        super().__init__(model_name)
    
    async def extract_ocr_text(self, image_path: str) -> Dict[str, Any]:
        """
        Extract pure OCR text from an image using Claude Vision API.
        
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
        
        if hasattr(response, 'content') and len(response.content) > 0:
            # Extract text from response
            text = response.content[0].text
            return {"text": text}
        else:
            return {"error": "Failed to extract OCR text", "response": str(response)}
    
    async def extract_structured_data(self, image_path: str) -> Dict[str, Any]:
        """
        Extract structured seed packet data from an image using Claude Vision API.
        The data structure matches the application's Seed model schema.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing the structured seed packet data
        """
        prompt = """
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
        
        response = await self.call_api_with_retry(image_path, prompt)
        
        if hasattr(response, 'content') and len(response.content) > 0:
            # Extract JSON from response
            content_text = response.content[0].text
            try:
                # Find JSON object in the response text (Claude might add text around the JSON)
                json_str = content_text
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                    
                structured_data = json.loads(json_str)
                return structured_data
            except json.JSONDecodeError as e:
                return {
                    "error": f"Failed to parse JSON response: {str(e)}",
                    "raw_response": content_text
                }
        else:
            return {"error": "Failed to extract structured data", "response": str(response)}

async def main():
    """
    Run a test of the Claude Vision API on seed packet images in the uploads directory.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Claude Vision API with seed packet images')
    parser.add_argument('model_name', nargs='?', default="claude-3-5-haiku-20241022",
                        help='Claude model name to use (default: claude-3-5-haiku-20241022)')
    args = parser.parse_args()
    
    model_name = args.model_name
    print(f"Using Claude model: {model_name}")
    
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
    
    # Initialize the Claude Vision tester with the specified model
    vision_tester = ClaudeVisionTester(model_name)
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"\n{'='*60}")
        print(f"Processing image {i+1}/{len(image_files)}: {image_path.name}")
        print(f"{'='*60}")
        
        # Extract OCR text
        print("\n1. Extracting raw OCR text using Claude Vision API...")
        try:
            ocr_result = await vision_tester.extract_ocr_text(str(image_path))
            
            if "error" not in ocr_result:
                print("\nExtracted OCR text:")
                print("-"*50)
                extracted_text = ocr_result["text"]
                # Print first 500 chars with ellipsis if longer
                print(extracted_text[:500] + ("..." if len(extracted_text) > 500 else ""))
                print("-"*50)
            else:
                print(f"\nError extracting OCR text: {ocr_result['error']}")
        except Exception as e:
            print(f"\nException during OCR extraction: {str(e)}")
        
        # Extract structured data
        print("\n2. Extracting structured data using Claude Vision API...")
        try:
            structured_data = await vision_tester.extract_structured_data(str(image_path))
            
            if "error" not in structured_data:
                print("\nExtracted structured data:")
                print("-"*50)
                print(json.dumps(structured_data, indent=2))
                print("-"*50)
                
                # Print data validation information
                missing_fields = [field for field in ["name", "variety", "maturity", "seed_depth", "spacing"] 
                                  if field not in structured_data or structured_data[field] is None]
                if missing_fields:
                    print("\nMissing important fields:", ", ".join(missing_fields))
                else:
                    print("\nAll key fields were extracted successfully.")
            else:
                print(f"\nError extracting structured data: {structured_data['error']}")
                if 'raw_response' in structured_data:
                    print("\nRaw response from Claude:")
                    print(structured_data['raw_response'][:500] + "..." 
                          if len(structured_data['raw_response']) > 500 else structured_data['raw_response'])
        except Exception as e:
            print(f"\nException during structured data extraction: {str(e)}")
            
        # Wait between images to avoid rate limits
        if i < len(image_files) - 1:
            print("\nWaiting 2 seconds before processing next image...")
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())