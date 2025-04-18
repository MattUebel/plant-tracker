# Technical Document: Extracting Seed Packet Data from Images using Gemini 2.5 Pro Experimental API

**Version:** 1.0
**Date:** 2025-03-29
**Target Audience:** Developers with Python experience.

## 1. Introduction

This document outlines the process for leveraging Google's advanced multimodal capabilities, specifically using the **`gemini-2.5-pro-preview-03-25`** model, to extract structured information from images of garden seed packets. Given your background in tech and security, you'll appreciate how this technique can automate data entry and analysis from visual sources.

The core approach involves sending both the image and a carefully crafted text prompt (requesting specific data fields in a structured format like JSON) to the Gemini API via the `google-generativeai` Python SDK.

## 2. Prerequisites

* **Python:** Version 3.x installed.
* **Google AI API Key:** Obtain an API key from [Google AI Studio](https://aistudio.google.com/). Ensure you have access enabled for the experimental models if required. *Alternatively, this process can be adapted for Vertex AI on Google Cloud with appropriate authentication.*
* **`google-generativeai` Library:** The Python SDK for the Gemini API.

## 3. Installation

Install the necessary Python library using pip:

```bash
pip install -q -U google-generativeai Pillow
```
*(Pillow is included for image handling).*

## 4. Authentication & Setup

Securely configure the SDK with your API key. Using environment variables is recommended.

```python
import os
import json
from google import generativeai
from google.generativeai import types
from PIL import Image # For loading image

# --- Configure API Key ---
# Best practice: Store API key in an environment variable
API_KEY = os.environ.get("GEMINI_API_KEY")

# Or uncomment and replace for quick testing (less secure):
# API_KEY = "YOUR_API_KEY"

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set or key is missing.")

# Initialize the Google AI Client
try:
    client = generativeai.Client(api_key=API_KEY)
    print("Google AI SDK Client Initialized.")
except Exception as e:
    print(f"Error initializing Google AI Client: {e}")
    exit()

# --- Specify the Model ---
# Using the requested experimental model
model_name = "gemini-2.5-pro-preview-03-25"
print(f"Target Model: {model_name}")

```

## 5. Core Process Overview

1.  **Load Image:** Read the target seed packet image file into memory (e.g., using Pillow).
2.  **Craft Prompt:** Create a detailed text prompt specifying exactly what information to extract and the desired output structure (JSON is recommended). This is the most critical step for accuracy.
3.  **Prepare Request:** Combine the prompt text and the image data into a list suitable for the API call.
4.  **Send API Request:** Call the Gemini API using the `generate_content` method of the model object, passing the combined prompt and image.
5.  **Process Response:** Receive the model's text response and parse the structured data (e.g., load the JSON string into a Python dictionary).

## 6. Crafting the Prompt for Seed Packets

The effectiveness of structured data extraction heavily relies on the prompt. It must clearly instruct the model. For seed packets, the prompt should:

* Identify the image type (seed packet).
* List the specific fields to extract (plant name, variety, days to germination, etc.).
* Define the exact output format (e.g., JSON object with specific keys).
* Instruct how to handle missing information (e.g., use `null`).
* Request only the structured data, without extra conversational text.

**Example Prompt:**

> ```text
> Analyze the provided image of a garden seed packet.
> Extract the following plant and seed information:
>
> - The common plant name (e.g., Tomato, Carrot, Zinnia)
> - The specific variety name (e.g., Beefsteak, Danvers 126, California Giant)
> - Days to Germination (e.g., "7-14 days")
> - Days to Maturity or Harvest (e.g., "75 days", "Approx. 60 days")
> - Recommended planting depth (e.g., "1/4 inch", "6mm")
> - Recommended seed spacing within the row (e.g., "6 inches apart", "Thin to 4 inches")
> - Recommended row spacing (e.g., "18 inches", "2 feet apart")
> - Light requirements (e.g., "Full Sun", "Partial Shade", "Full Sun to Light Shade")
> - Basic planting instructions or key tips (Summarize if long)
> - Net weight or approximate seed count (e.g., "Net Wt. 500mg", "Approx 50 seeds")
> - Recommended USDA Planting Zones (e.g., "Zones 3-9", "Annual for all zones")
>
> Please provide the output strictly as a JSON object using the following keys:
> `plant_name`, `variety`, `days_to_germination`, `days_to_maturity`, `planting_depth`, `seed_spacing`, `row_spacing`, `light_requirement`, `planting_instructions`, `net_weight_or_count`, `usda_zone`.
>
> If a specific piece of information cannot be clearly identified on the packet, use `null` as the value for that key.
> Extract the text as accurately as possible from the image.
> Do not include any introductory text, explanations, or markdown formatting outside the JSON object itself.
> ```

## 7. Example Python Implementation

This script combines the steps to extract data from a seed packet image using the specified model.

```python
import os
import json
from google import generativeai
from google.generativeai import types
from PIL import Image

# --- 1. Authentication & Setup ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not