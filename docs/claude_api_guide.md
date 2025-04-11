```markdown
# Implementing a FastAPI Application for Extracting Plant Information from Seed Packet Images

This guide provides a step-by-step approach to building a FastAPI application that extracts primary plant information from images of seed packets. The extracted data is structured in JSON format, facilitating efficient tracking and management of a home garden.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Project Setup](#project-setup)
4. [Building the FastAPI Application](#building-the-fastapi-application)
   - [1. Import Necessary Modules](#1-import-necessary-modules)
   - [2. Initialize the FastAPI App](#2-initialize-the-fastapi-app)
   - [3. Define the Claude API Client](#3-define-the-claude-api-client)
   - [4. Create the Image Upload Endpoint](#4-create-the-image-upload-endpoint)
   - [5. Run the Application](#5-run-the-application)
5. [Testing the Application](#testing-the-application)
6. [Conclusion](#conclusion)
7. [References](#references)

## Introduction

Managing a home garden effectively involves keeping track of various plant species, their planting requirements, and growth cycles. Seed packets often contain essential information such as plant names, types, planting instructions, spacing, sun exposure, and germination times. By digitizing and extracting this information into a structured JSON format, gardeners can maintain an organized database to enhance garden planning and maintenance.

This guide demonstrates how to develop a FastAPI application that accepts images of seed packets, processes them using Anthropic's Claude AI model to extract relevant plant information, and returns the data in JSON format.

## Prerequisites

Before proceeding, ensure you have the following:

- **Python 3.7+**: FastAPI requires Python version 3.7 or higher.
- **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python.
- **Uvicorn**: A lightning-fast ASGI server implementation, using `uvloop` and `httptools`.
- **Anthropic's Python SDK**: To interact with Claude AI's API.
- **Pillow**: A Python Imaging Library (PIL) fork that supports opening, manipulating, and saving image files.
- **Requests**: A simple HTTP library for Python.

Install the necessary packages using the following commands:

```bash
pip install fastapi uvicorn anthropic pillow requests python-multipart
```

## Project Setup

1. **Create a Project Directory**: Organize your project files in a dedicated directory.

   ```bash
   mkdir seed_packet_extractor
   cd seed_packet_extractor
   ```

2. **Set Up Environment Variables**: Store sensitive information like API keys securely using environment variables.

   - On Linux/macOS:

     ```bash
     export ANTHROPIC_API_KEY='your_anthropic_api_key'
     ```

   - On Windows:

     ```bash
     set ANTHROPIC_API_KEY='your_anthropic_api_key'
     ```

   Alternatively, use a `.env` file to load environment variables.

## Building the FastAPI Application

### 1. Import Necessary Modules

Begin by importing the required modules:

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from anthropic import Anthropic
from PIL import Image
import base64
import os
import io
```

### 2. Initialize the FastAPI App

Create an instance of the FastAPI application:

```python
app = FastAPI(title="Seed Packet Information Extractor")
```

### 3. Define the Claude API Client

Initialize the Claude AI client using the API key:

```python
claude_api_key = os.getenv("ANTHROPIC_API_KEY")
if not claude_api_key:
    raise ValueError("Anthropic API key is not set. Please set the ANTHROPIC_API_KEY environment variable.")

client = Anthropic(api_key=claude_api_key)
```

### 4. Create the Image Upload Endpoint

Define an endpoint to handle image uploads and process them to extract plant information:

```python
@app.post("/extract-plant-info/")
async def extract_plant_info(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")

    # Read and encode the image
    image_data = await file.read()
    encoded_image = base64.b64encode(image_data).decode()

    # Prepare the Claude AI prompt
    prompt = (
        "Extract structured plant information from the provided seed packet image. "
        "Return a JSON object with the following fields: 'plant_name', 'plant_type', "
        "'days_to_germinate', 'days_to_harvest', 'plant_spacing', 'planting_depth', "
        "'sun_exposure', 'watering_instructions', and 'description'. If any data is unavailable, use null for that field."
    )

    # Send the request to Claude AI
    response = client.completions.create(
        model="claude-3-5-haiku-20241022",
        max_tokens_to_sample=1024,
        prompt=prompt,
        images=[{"image": encoded_image}]
    )

    # Parse the response
    try:
        plant_info = response["completion"].strip()
        plant_info_json = json.loads(plant_info)
    except (KeyError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Error processing Claude AI response: {e}")

    return JSONResponse(content=plant_info_json)
```

### 5. Run the Application

Use Uvicorn to run the FastAPI application:

```bash
uvicorn main:app --reload
```

Replace `main` with the name of your Python file if it's different.

## Testing the Application

1. **Access the Interactive API Docs**: FastAPI provides an interactive Swagger UI at `http://127.0.0.1:8000/docs`. Here, you can test the `/extract-plant-info/` endpoint by uploading an image of a seed packet.

2. **Upload an Image**: Use the provided interface to upload a JPEG or PNG image of a seed packet.

3. **View the Extracted Information**: Upon successful processing, the API will return a JSON object containing the extracted plant information.

## Conclusion

By following this guide, you've developed a FastAPI application capable of extracting structured plant information from seed packet images using Anthropic's Claude AI model. This tool can significantly aid in organizing and managing your home garden by digitizing essential planting information.

## References

- [FastAPI: Request Files](https://fastapi.tiangolo.com/tutorial/request-files/)
- [Anthropic Claude AI Documentation](https://docs.anthropic.com/claude-docs)
- [Pillow (PIL Fork) Documentation](https://pillow.readthedocs.io/en/stable/)
- [Building a File Upload and Download API with Python and FastAPI](https://medium.com/@chodvadiyasaurabh/building-a-file-upload-and-download-api-with-python-and-fastapi-3de94e4d 