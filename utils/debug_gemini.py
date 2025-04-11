#!/usr/bin/env python
"""
Simplified debugging script for Gemini Vision API
Works within Docker container environment
"""
import sys
import os
import traceback
import json
from PIL import Image

# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gemini_debug")

# Path to the image you want to test
IMAGE_PATH = (
    sys.argv[1] if len(sys.argv) > 1 else "uploads/seed_19_20250408182131_221b23fa.jpg"
)

try:
    print(f"Debug script running - Python {sys.version}")
    print(f"Testing image: {IMAGE_PATH}")

    # Import and configure Gemini API
    try:
        import google.generativeai as genai

        print("Successfully imported google.generativeai")

        # Get API key from environment
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("No GEMINI_API_KEY found in environment")
            sys.exit(1)

        genai.configure(api_key=api_key)
        print("Configured Gemini API with key from environment")

        # Create model instance
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")
        model = genai.GenerativeModel(model_name)
        print(f"Created model instance with: {model_name}")

    except ImportError as e:
        print(f"Error importing google.generativeai: {e}")
        sys.exit(1)

    # Define JSON schema for structured extraction
    schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The basic seed/plant type (e.g., Tomato, Basil)",
            },
            "variety": {
                "type": ["string", "null"],
                "description": "The specific variety/cultivar name",
            },
            "ocr_text": {
                "type": "string",
                "description": "Extracted text from the image",
            },
        },
        "required": ["name", "ocr_text"],
    }

    # Simple prompt
    prompt = "What type of seed packet is this? Please extract the readable text and key information."

    # Load image
    try:
        img = Image.open(IMAGE_PATH)
        print(f"Image loaded successfully: {img.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        traceback.print_exc()
        sys.exit(1)

    # First, try without JSON schema to see raw response
    print("\n--- Testing simple response ---")
    try:
        response = model.generate_content([prompt, img])

        print("Response type:", type(response))
        print("Has text attribute:", hasattr(response, "text"))
        print("Has parts:", hasattr(response, "parts"))
        print("Has candidates:", hasattr(response, "candidates"))

        if hasattr(response, "text"):
            print("\nResponse text:")
            print(
                response.text[:500] + "..."
                if len(response.text) > 500
                else response.text
            )

        if hasattr(response, "candidates"):
            print("\nCandidate information:")
            for i, candidate in enumerate(response.candidates):
                print(
                    f"  Candidate {i} finish reason: {getattr(candidate, 'finish_reason', 'unknown')}"
                )
                print(
                    f"  Candidate {i} safety ratings: {getattr(candidate, 'safety_ratings', [])}"
                )

    except Exception as e:
        print(f"Error with simple response: {e}")
        traceback.print_exc()

    # Next, try with JSON schema
    print("\n--- Testing with JSON schema ---")
    try:
        schema_response = model.generate_content(
            [prompt, img], generation_config={"response_schema": schema}
        )

        print("Schema response type:", type(schema_response))
        print("Has text attribute:", hasattr(schema_response, "text"))

        if hasattr(schema_response, "text"):
            print("\nSchema response text:")
            print(schema_response.text)

            try:
                data = json.loads(schema_response.text)
                print("\nParsed JSON data:")
                print(json.dumps(data, indent=2))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")

        if hasattr(schema_response, "candidates"):
            print("\nCandidate information with schema:")
            for i, candidate in enumerate(schema_response.candidates):
                print(
                    f"  Candidate {i} finish reason: {getattr(candidate, 'finish_reason', 'unknown')}"
                )
                print(
                    f"  Candidate {i} safety ratings: {getattr(candidate, 'safety_ratings', [])}"
                )
                if hasattr(candidate, "content") and hasattr(
                    candidate.content, "parts"
                ):
                    for j, part in enumerate(candidate.content.parts):
                        print(f"  Part {j} type: {type(part)}")
                        if hasattr(part, "text"):
                            print(
                                f"  Part {j} text: {part.text[:100]}..."
                                if len(part.text) > 100
                                else part.text
                            )

    except Exception as e:
        print(f"Error with schema response: {e}")
        traceback.print_exc()

except Exception as e:
    print(f"General error: {e}")
    traceback.print_exc()
