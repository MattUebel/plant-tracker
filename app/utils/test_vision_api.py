"""
Utility script to test Mistral Vision and OCR APIs with seed packet images.
This helps us understand the API behavior and optimize our OCR implementation.
Usage:
    cd /Users/mattuebel/mattuebel/plant-tracker
    python -m app.utils.test_vision_api
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

# Import after path setup
from dotenv import load_dotenv

# Load environment variables from project root
env_path = Path(__file__).parent.parent.parent / '.env'
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
        # This improves OCR by making text more distinct from background
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 4. Denoise to remove specks while preserving text
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        # 5. Correct skew if needed
        # Skew correction code can be added here if text alignment issues are detected
        
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
            raise ValueError("MISTRAL_API_KEY environment variable not set")
    
    async def call_api_with_retry(self, 
                           api_func: Callable, 
                           max_retries: int = 3, 
                           initial_delay: float = 2.0) -> Dict[str, Any]:
        """
        Call an API function with exponential backoff retry logic.
        
        Args:
            api_func: Async function that makes the actual API call
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            
        Returns:
            API response as dictionary
        """
        retries = 0
        delay = initial_delay
        
        while True:
            try:
                return await api_func()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit exceeded
                    if retries >= max_retries:
                        raise ValueError(f"API rate limit exceeded after {max_retries} retries")
                    
                    print(f"Rate limit exceeded. Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                    retries += 1
                elif e.response.status_code == 401:
                    raise ValueError("Invalid API key or authentication error")
                else:
                    raise ValueError(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                raise ValueError(f"API call failed: {str(e)}")


class MistralOCRTester(MistralAPICaller):
    """Test Mistral's dedicated OCR API for text extraction."""
    
    def __init__(self):
        super().__init__()
        self.api_endpoint = "https://api.mistral.ai/v1/ocr"
    
    async def extract_text(self, image_path: str, use_preprocessing: bool = True) -> Dict[str, Any]:
        """
        Extract text from an image using Mistral OCR API.
        
        Args:
            image_path: Path to the image file
            use_preprocessing: Whether to apply image preprocessing
            
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
    
    async def test_vision_api(self, 
                       image_path: str, 
                       prompt: str,
                       model: str = "mistral-large",
                       json_output: bool = False,
                       max_tokens: int = 4000) -> Dict[str, Any]:
        """
        Test Mistral Vision API with a single image and prompt.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt to send with the image
            model: Model name to use
            json_output: Whether to request JSON formatted output
            max_tokens: Maximum tokens in the response
            
        Returns:
            Vision API response
        """
        async def make_vision_request():
            # Read and encode the image
            with open(image_path, "rb") as f:
                image_data = f.read()
                image_b64 = base64.b64encode(image_data).decode("utf-8")
                
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare the request payload
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1
            }
            
            if json_output:
                payload["response_format"] = {"type": "json_object"}
            
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
        return await self.call_api_with_retry(make_vision_request)
    
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
    
    async def test_structured_extraction_pipeline(self, image_path: str) -> Dict[str, Any]:
        """
        Test a complete pipeline using OCR first, then structured extraction.
        
        This is the two-step approach recommended in the Mistral OCR guide:
        1. Extract text using dedicated OCR API
        2. Structure the text using Chat Completion API with a schema
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Structured data extracted from the image
        """
        print(f"\nAnalyzing image: {Path(image_path).name}")
        print("Step 1: Running OCR to extract text...")
        
        # First use OCR API to extract text
        ocr_tester = MistralOCRTester()
        ocr_result = await ocr_tester.extract_text(image_path)
        
        # Extract text from OCR result
        extracted_text = ""
        if "pages" in ocr_result:
            for page in ocr_result["pages"]:
                if "markdown" in page:
                    extracted_text += page["markdown"] + "\n"
        
        if not extracted_text.strip():
            print("OCR did not extract any text from the image.")
            return {"error": "No text extracted"}
        
        print(f"OCR extracted {len(extracted_text)} characters of text")
        
        # Define the schema based on seed packet information
        seed_packet_schema = {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string", 
                    "description": "The name of the seed product"
                },
                "brand": {
                    "type": "string", 
                    "description": "The brand of the seed"
                },
                "variety": {
                    "type": "string", 
                    "description": "The specific variety of the plant"
                },
                "net_weight": {
                    "type": "string", 
                    "description": "The net weight of the seeds"
                },
                "sowing_instructions": {
                    "type": "string", 
                    "description": "Instructions for sowing the seeds"
                },
                "planting_depth": {
                    "type": "string", 
                    "description": "The recommended planting depth"
                },
                "spacing": {
                    "type": "string", 
                    "description": "The recommended spacing between plants"
                },
                "days_to_maturity": {
                    "type": "string", 
                    "description": "The number of days until maturity"
                },
                "germination_rate": {
                    "type": "string",
                    "description": "Expected germination rate if specified"
                },
                "plant_height": {
                    "type": "string",
                    "description": "Expected height of mature plant"
                },
                "light_requirements": {
                    "type": "string",
                    "description": "Sunlight requirements (full sun, partial shade, etc.)"
                },
                "watering_needs": {
                    "type": "string",
                    "description": "Watering instructions or requirements"
                },
                "harvest_instructions": {
                    "type": "string",
                    "description": "Instructions for when and how to harvest"
                }
            },
            "required": ["product_name", "brand", "variety"]
        }
        
        print("Step 2: Extracting structured data from OCR text...")
        # Use the extracted text to get structured data
        structured_data = await self.extract_structured_data_from_ocr(extracted_text, seed_packet_schema)
        
        return structured_data
    
    async def test_structured_extraction(self, image_path: str) -> None:
        """Test different approaches to structured data extraction."""
        
        print(f"\nAnalyzing image: {Path(image_path).name}")
        
        # First approach: Direct vision API with comprehensive prompt
        print("\nApproach 1: Testing direct vision API extraction...")
        comprehensive_prompt = """
        Analyze this seed packet image and extract the following information in a structured JSON format. Be precise and only extract information that is clearly visible in the image:
        - name: Basic type of seed (e.g., "Tomato")
        - variety: Specific variety name if present
        - brand: Brand or manufacturer name if visible
        - germination_rate: As decimal (e.g., 0.85 for 85%) if specified
        - maturity: Days to maturity as integer if specified
        - growth: Growth habit (e.g., "Determinate", "Vining") if specified
        - seed_depth: Planting depth in inches (decimal) if specified
        - spacing: Plant spacing in inches (decimal) if specified
        - planting_instructions: Full planting instructions if visible
        - description: Any description of the plant/variety
        - notes: Any additional important information
        You must return a valid JSON object containing these fields. Use null for any information not present in the image. Do not make assumptions or fill in missing data.
        """
        
        direct_vision_result = await self.test_vision_api(
            image_path, 
            comprehensive_prompt,
            json_output=True
        )
        
        if direct_vision_result and 'choices' in direct_vision_result:
            content = direct_vision_result['choices'][0]['message']['content']
            try:
                parsed_direct = json.loads(content)
                print("\nDirect Vision API Results (prettified):")
                print(json.dumps(parsed_direct, indent=2))
            except json.JSONDecodeError:
                print("\nWarning: Direct Vision API response was not valid JSON:")
                print(content)
        
        # Second approach: OCR then structured extraction pipeline
        print("\nApproach 2: Testing OCR + structured extraction pipeline...")
        pipeline_result = await self.test_structured_extraction_pipeline(image_path)
        
        print("\nOCR + Structured Extraction Pipeline Results (prettified):")
        print(json.dumps(pipeline_result, indent=2))
        
        # Compare approaches if both worked
        if ('choices' in direct_vision_result and 
            'error' not in pipeline_result):
            print("\nComparing approaches...")
            # Here you could implement comparison metrics


async def main():
    print("Starting Mistral API Tests for Seed Packet Analysis")
    
    # Initialize tester
    try:
        vision_tester = MistralVisionTester()
    except ValueError as e:
        print(f"Error: {e}")
        return
        
    # Find test images
    root_dir = Path(__file__).parent.parent.parent
    uploads_dir = root_dir / "uploads"
    
    # Look for test images in uploads directory
    image_patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(list(uploads_dir.glob(pattern)))
    
    if not image_files:
        print(f"No image files found in {uploads_dir}")
        return
        
    print(f"\nFound {len(image_files)} images to analyze")
    
    # Process each image
    for image_path in image_files:
        print(f"\n{'='*80}")
        try:
            # Test structured extraction with different approaches
            await vision_tester.test_structured_extraction(str(image_path))
            print("\nWaiting 3 seconds before next test to avoid rate limiting...")
            await asyncio.sleep(3)  # Enhanced rate limiting
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            # Continue with next image
            continue


if __name__ == "__main__":
    asyncio.run(main())