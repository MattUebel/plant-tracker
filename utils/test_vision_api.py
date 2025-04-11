"""
Utility script to test Mistral OCR API with seed packet images.
This helps us understand the API behavior and optimize our OCR implementation.

Usage:
    cd /Users/mattuebel/mattuebel/plant-tracker
    python -m utils.test_vision_api
"""
import os
import sys
import base64
import httpx
import json
import asyncio
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Load environment variables from project root
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)


class ImagePreprocessor:
    """Image preprocessing utilities to improve OCR accuracy."""
    
    @staticmethod
    def preprocess_image(image_path: str) -> np.ndarray:
        """Apply preprocessing steps to improve OCR quality."""
        # Read the image
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Failed to read image at {image_path}")

        # Apply preprocessing steps
        # 1. Resize to maintain resolution but ensure size is reasonable for API
        max_dim = 1800
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        # 2. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Apply adaptive thresholding for better text contrast
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 4. Denoise to remove specks while preserving text
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        # 5. Correct skew if needed (simplifying this step for now)
        
        return denoised

    @staticmethod
    def encode_image(image: Union[str, np.ndarray], format_type: str = "jpeg") -> str:
        """
        Encode an image to base64.
        
        Args:
            image: Either a path to an image file or a numpy array
            format_type: The format to encode (jpeg, png)
        
        Returns:
            Base64 encoded string
        """
        if isinstance(image, str):
            with open(image, "rb") as f:
                image_data = f.read()
                return base64.b64encode(image_data).decode("utf-8")
        else:
            # For numpy array
            success, buffer = cv2.imencode(f'.{format_type}', image)
            if not success:
                raise ValueError("Failed to encode image")
            return base64.b64encode(bytes(buffer)).decode("utf-8")


class MistralAPICaller:
    """Base class for Mistral API calls with retry logic."""
    
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
        
        self.api_endpoint = "https://api.mistral.ai/v1/chat/completions"
        self.max_retries = 3
        self.initial_retry_delay = 1.0
    
    async def call_api_with_retry(self, api_func: Callable) -> Dict[str, Any]:
        """
        Call an API function with exponential backoff retry logic.
        
        Args:
            api_func: Async function that makes the API call
            
        Returns:
            API response as a dictionary
        """
        retries = 0
        delay = self.initial_retry_delay
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                return await api_func()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit exceeded
                    print(f"Rate limit exceeded. Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                    retries += 1
                    last_exception = e
                else:
                    print(f"API error: {e.response.status_code} - {e.response.text}")
                    raise
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                raise
                
        if last_exception:
            raise last_exception
        
        return {"error": "Max retries exceeded"}


class MistralOCRTester(MistralAPICaller):
    """Test Mistral's dedicated OCR API for text extraction."""
    
    def __init__(self):
        super().__init__()
        self.api_endpoint = "https://api.mistral.ai/v1/ocr"
    
    async def extract_text(self, image_path: str, use_preprocessing: bool = False) -> Dict[str, Any]:
        """
        Extract text from an image using Mistral OCR API.
        
        Args:
            image_path: Path to the image file
            use_preprocessing: Whether to apply image preprocessing (defaults to False now)
            
        Returns:
            OCR API response containing extracted text
        """
        preprocessor = ImagePreprocessor()
        
        # Define the API call function
        async def make_ocr_request():
            if use_preprocessing:
                # Preprocess and encode the image
                processed_image = preprocessor.preprocess_image(image_path)
                image_b64 = preprocessor.encode_image(processed_image)
            else:
                # Just encode the original image
                image_b64 = preprocessor.encode_image(image_path)
                
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "mistral-ocr-latest",
                "document": {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_b64}"
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
        
        # Call API with retry logic
        return await self.call_api_with_retry(make_ocr_request)


class MistralVisionTester(MistralAPICaller):
    """Test Mistral Vision API for image understanding and structured data extraction."""
    
    def __init__(self):
        super().__init__()
        self.api_endpoint = "https://api.mistral.ai/v1/chat/completions"
    
    async def extract_structured_data_from_ocr(self, 
                                     ocr_text: str, 
                                     schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from OCR text using Mistral's Chat Completion API.
        
        Args:
            ocr_text: The text extracted by OCR
            schema: JSON schema defining the structure of the data to extract
            
        Returns:
            Structured data extracted from the OCR text
        """
        async def make_structured_extraction_request():
            # Construct a prompt that uses the OCR text and schema
            prompt = f"""
            I have extracted the following text from a seed packet image using OCR:
            
            {ocr_text}
            
            Please extract structured information according to this JSON schema:
            
            {json.dumps(schema, indent=2)}
            
            Return ONLY a valid JSON object following this schema. Use null for any fields where information is not present in the text.
            """
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "mistral-small-latest",  # Can use small model for structured extraction
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
        
        # Call API with retry logic
        result = await self.call_api_with_retry(make_structured_extraction_request)
        
        if result and 'choices' in result:
            content = result['choices'][0]['message']['content']
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print("\nWarning: Response was not valid JSON:")
                print(content)
                return {"error": "Failed to parse JSON response"}
        
        return {"error": "No valid response from API"}


async def main():
    """Run a test of the Mistral OCR API on sample seed packet images."""
    # Define the schema for seed packet data
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The name of the seed (e.g., Tomato, Pepper)"},
            "variety": {"type": "string", "description": "The specific variety of the seed"},
            "germination_rate": {"type": "number", "description": "The germination rate as a decimal (e.g., 0.85 for 85%)"},
            "maturity": {"type": "integer", "description": "Days to maturity"},
            "growth": {"type": "string", "description": "Growth habit (e.g., Determinate, Indeterminate, Bush, Vining)"},
            "seed_depth": {"type": "number", "description": "Planting depth in inches"},
            "spacing": {"type": "number", "description": "Spacing between plants in inches"},
            "notes": {"type": "string", "description": "Additional information from the packet"}
        }
    }
    
    # Find available images in the uploads directory
    uploads_dir = Path(__file__).parent.parent / "uploads"
    if not uploads_dir.exists():
        print(f"Uploads directory not found: {uploads_dir}")
        return
    
    # List image files
    image_files = [f for f in uploads_dir.glob("*.jpg") if f.is_file()]
    image_files.extend([f for f in uploads_dir.glob("*.JPG") if f.is_file()])
    image_files.extend([f for f in uploads_dir.glob("*.jpeg") if f.is_file()])
    image_files.extend([f for f in uploads_dir.glob("*.JPEG") if f.is_file()])
    image_files.extend([f for f in uploads_dir.glob("*.png") if f.is_file()])
    image_files.extend([f for f in uploads_dir.glob("*.PNG") if f.is_file()])
    
    if not image_files:
        print("No images found in the uploads directory.")
        return
    
    print(f"Found {len(image_files)} image(s) to process.")
    
    # Initialize the OCR tester
    ocr_tester = MistralOCRTester()
    vision_tester = MistralVisionTester()
    
    # Process each image - test both with and without preprocessing
    for i, image_path in enumerate(image_files):
        print(f"\n--- Processing image {i+1}/{len(image_files)}: {image_path.name} ---")
        
        # Test original image without preprocessing
        print("\n1. Extracting text from ORIGINAL image using Mistral OCR API...")
        ocr_result_original = await ocr_tester.extract_text(str(image_path), use_preprocessing=False)
        
        if 'pages' in ocr_result_original:
            # Extract text from OCR result
            extracted_text_original = ""
            for page in ocr_result_original['pages']:
                extracted_text_original += page.get('markdown', '')
            
            print("\nExtracted text from ORIGINAL image:")
            print("="*50)
            print(extracted_text_original[:500] + ("..." if len(extracted_text_original) > 500 else ""))
            print("="*50)
            
            print("\n2. Extracting structured data from ORIGINAL image OCR text...")
            structured_data_original = await vision_tester.extract_structured_data_from_ocr(extracted_text_original, schema)
            
            print("\nStructured data from ORIGINAL image:")
            print("="*50)
            print(json.dumps(structured_data_original, indent=2))
            print("="*50)
        else:
            print("\nError: OCR result for ORIGINAL image does not contain 'pages' field.")
            print(f"OCR result: {ocr_result_original}")
        
        # Test with preprocessing
        print("\n3. Extracting text from PREPROCESSED image using Mistral OCR API...")
        ocr_result_preprocessed = await ocr_tester.extract_text(str(image_path), use_preprocessing=True)
        
        if 'pages' in ocr_result_preprocessed:
            # Extract text from OCR result
            extracted_text_preprocessed = ""
            for page in ocr_result_preprocessed['pages']:
                extracted_text_preprocessed += page.get('markdown', '')
            
            print("\nExtracted text from PREPROCESSED image:")
            print("="*50)
            print(extracted_text_preprocessed[:500] + ("..." if len(extracted_text_preprocessed) > 500 else ""))
            print("="*50)
            
            print("\n4. Extracting structured data from PREPROCESSED image OCR text...")
            structured_data_preprocessed = await vision_tester.extract_structured_data_from_ocr(extracted_text_preprocessed, schema)
            
            print("\nStructured data from PREPROCESSED image:")
            print("="*50)
            print(json.dumps(structured_data_preprocessed, indent=2))
            print("="*50)
            
            # Compare results
            print("\nCOMPARISON OF TEXT EXTRACTION:")
            print(f"Original text length: {len(extracted_text_original)}")
            print(f"Preprocessed text length: {len(extracted_text_preprocessed)}")
            
            if len(extracted_text_original.strip()) > len(extracted_text_preprocessed.strip()):
                print("RESULT: Original image yielded more text")
            elif len(extracted_text_original.strip()) < len(extracted_text_preprocessed.strip()):
                print("RESULT: Preprocessed image yielded more text")
            else:
                print("RESULT: Both methods yielded similar amount of text")
        else:
            print("\nError: OCR result for PREPROCESSED image does not contain 'pages' field.")
            print(f"OCR result: {ocr_result_preprocessed}")
        
        if i < len(image_files) - 1:
            print("\nWaiting 2 seconds before processing next image...")
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())