Extracting Structured Data from Images Using Mistral AI: A Comprehensive Guide
The increasing volume of visual information necessitates efficient methods for extracting and organizing data contained within images. This report details how Mistral AI's Optical Character Recognition (OCR) and Chat Completion APIs can be leveraged programmatically using Python to extract structured data from images, such as product labels or fact sheets. The process involves utilizing the OCR API to transcribe text from the image and subsequently employing the Chat Completion API, guided by a predefined schema, to structure the extracted information into a usable format.
Leveraging Mistral AI's OCR API for Text Extraction
Mistral AI offers a cutting-edge OCR API designed to accurately interpret and extract text from various document types, including images 1. This API goes beyond simple character recognition by preserving document structure, handling complex layouts like tables and multi-column text, and even understanding mathematical expressions 1. The API can process documents provided as URLs or uploaded files, supporting formats like PDF and various image types 3. The output from the OCR API is typically provided in Markdown format, facilitating easy parsing and further processing 2. While specific examples of extracting information from product labels are not explicitly detailed in the provided materials, the described capabilities strongly suggest its applicability to such tasks, including processing images of seed packets to extract relevant information 2. The OCR API is accessed using the mistral-ocr-latest model 3.
Structuring Extracted Data with Mistral AI's Chat Completion API
Once the text has been extracted from an image using the OCR API, the Mistral AI Chat Completion API can be employed to structure this raw text into a desired format 8. This API utilizes powerful language models that can understand instructions and generate output in a structured manner, particularly in JSON format 7. By providing the extracted text as part of the prompt, along with a clear definition of the desired data schema, the Chat Completion API can intelligently parse the information and organize it according to the specified structure. This enables the transformation of unstructured text into machine-readable and easily accessible data.
Defining a Data Schema for Structured Extraction
A crucial step in extracting structured data is defining a clear and comprehensive data schema 15. This schema acts as a blueprint, guiding the Chat Completion API on how to organize the extracted information. The schema can be defined using JSON schema, a standard format for describing the structure of JSON data 15. It allows for specifying the names of the fields to be extracted, their data types (e.g., string, integer, boolean), and any constraints or descriptions to further guide the language model 16.
For example, to extract information from a garden seed packet, a schema might be defined as follows:

JSON


{
  "type": "object",
  "properties": {
    "product_name": { "type": "string", "description": "The name of the seed product" },
    "brand": { "type": "string", "description": "The brand of the seed" },
    "variety": { "type": "string", "description": "The specific variety of the plant" },
    "net_weight": { "type": "string", "description": "The net weight of the seeds" },
    "sowing_instructions": { "type": "string", "description": "Instructions for sowing the seeds" },
    "planting_depth": { "type": "string", "description": "The recommended planting depth" },
    "spacing": { "type": "string", "description": "The recommended spacing between plants" },
    "days_to_maturity": { "type": "string", "description": "The number of days until maturity" }
  },
  "required": ["product_name", "brand", "variety"]
}


This schema specifies the fields of interest and provides descriptions to help the Chat Completion model understand the type of information to look for in the extracted text 16.
Programmatic Implementation with Python
To automate the process of extracting structured data, the Mistral AI OCR and Chat Completion APIs can be used together programmatically with Python. This involves the following steps:
Install the Mistral AI Python Client: The official Python client library, mistralai, needs to be installed using pip 2.
Set up API Key: Obtain a Mistral AI API key from the La Plateforme developer suite and set it as an environment variable (MISTRAL_API_KEY) for secure access 2.
Initialize the Mistral Client: Create an instance of the Mistral client in your Python script, providing the API key 2.
Read the Image: Load the image file (e.g., a seed packet) into your Python script. This can be done using libraries like Pillow or OpenCV 20. For processing images directly, it might be necessary to encode the image in base64 format or provide a URL to the image 2.
Call the OCR API: Use the client.ocr.process method, specifying the mistral-ocr-latest model and providing the image data either as a URL or a base64 encoded string within the document parameter 3.
Extract Text from OCR Response: The response from the OCR API will contain the extracted text, typically within the pages attribute, where each page has a markdown attribute containing the transcribed text 2.
Define the Data Schema: Define the desired data schema in JSON format as a Python dictionary or a JSON string 15.
Construct the Chat Completion Prompt: Create a prompt for the Chat Completion API that includes the extracted text from the OCR response and clearly instructs the model to structure this information according to the defined schema. Specify that the output should be in JSON format 7.
Call the Chat Completion API: Use the client.chat.complete method, providing the desired chat model (e.g., mistral-small-latest or mistral-large-latest), the constructed prompt as a user message, and instructions for structured output 3.
Parse the Structured Output: The response from the Chat Completion API will contain the structured data, typically in JSON format within the content of the assistant's message. Parse this JSON data in your Python script for further use 7.
A basic Python code example illustrating this process:

Python


import os
from mistralai import Mistral
import base64
import json

# Initialize Mistral client
api_key = os.environ
client = Mistral(api_key=api_key)

# Define the data schema
schema = {
  "type": "object",
  "properties": {
    "product_name": {"type": "string"},
    "brand": {"type": "string"},
    "variety": {"type": "string"}
  },
  "required": ["product_name", "brand", "variety"]
}

# Path to the image file
image_path = "seed_packet.jpg"

# Encode the image to base64
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# Call the OCR API
ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
)

# Extract text from the OCR response
extracted_text = ""
if ocr_response.pages:
    for page in ocr_response.pages:
        extracted_text += page.markdown

# Construct the chat completion prompt
prompt = f"""Please extract the following information from this text according to the JSON schema below:

{extracted_text}

JSON Schema:
```json
{json.dumps(schema)}


Return the extracted information as a JSON object."""
Call the Chat Completion API
chat_response = client.chat.complete(
model="mistral-small-latest",
messages=[{"role": "user", "content": prompt}],
response_format={"type": "json_object"}
)
Parse and print the structured output
if chat_response.choices and chat_response.choices.message.content:
structured_data = json.loads(chat_response.choices.message.content)
print(json.dumps(structured_data, indent=2))
else:
print("No structured data extracted.")



## Image Preprocessing for Enhanced OCR Accuracy

The accuracy of OCR results can be significantly impacted by the quality of the input image [20, 21, 22, 23]. Therefore, applying image preprocessing techniques before sending the image to the OCR API is often beneficial. Common preprocessing steps include:

*   **Normalization:** Adjusting the pixel intensity range to a standard level [20].
*   **Skew Correction:** Straightening any tilt or skew in the image to ensure horizontal alignment of text [20, 22, 23].
*   **Image Scaling:** Resizing the image to an optimal resolution (typically above 300 DPI) for better character recognition [20, 22].
*   **Noise Removal:** Reducing speckles, smudges, or other artifacts that might interfere with OCR [20, 22, 23].
*   **Binarization:** Converting the image to black and white to enhance the contrast between text and background [20, 22, 23].
*   **Cropping:** Focusing the OCR on the relevant text areas by removing unnecessary borders or graphics [22].

Libraries like OpenCV and Pillow in Python provide functionalities to perform these image preprocessing steps [20]. The specific preprocessing techniques required will depend on the characteristics of the images being processed.

## Error Handling and Retry Mechanisms

When interacting with external APIs, it is essential to implement robust error handling and retry mechanisms [13, 19, 24, 25, 26]. Network issues, temporary server problems, or incorrect API usage can lead to errors. The Mistral AI Python client raises exceptions for API errors [6]. Common error codes include 401 (Unauthorized), 429 (Too Many Requests - indicating rate limiting), and 500 (Internal Server Error) [25, 27].

Implementing `try...except` blocks in Python can help catch these exceptions and handle them gracefully. For rate limiting errors (429), implementing a retry mechanism with exponential backoff can be effective. This involves waiting for an increasing amount of time before retrying the API call.

A simple example of a retry mechanism:

```python
import time
from mistralai import Mistral, APIError

def call_api_with_retry(client, method, max_retries=3, initial_delay=1):
    retries = 0
    delay = initial_delay
    while retries < max_retries:
        try:
            return method()
        except APIError as e:
            if e.status_code == 429:
                print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                retries += 1
            else:
                raise  # Re-raise other API errors
    raise Exception(f"Failed to call API after {max_retries} retries.")

# Example usage for OCR:
# ocr_response = call_api_with_retry(client, lambda: client.ocr.process(model="mistral-ocr-latest", document={"type": "image_url", "image_url": image_url}))

# Example usage for Chat Completion:
# chat_response = call_api_with_retry(client, lambda: client.chat.complete(model="mistral-small-latest", messages=[{"role": "user", "content": prompt}]))


Understanding Mistral AI API Rate Limits and Pricing
Mistral AI imposes rate limits on its APIs to ensure fair usage and prevent abuse 7. These limits are typically based on the number of requests per second (RPS) and the number of tokens processed per minute or month 28. The specific rate limits depend on the user's subscription tier and the model being used 27. It is crucial to consult the Mistral AI La Plateforme dashboard (console.mistral.ai/limits/) for the most up-to-date information on your account's rate limits 30. Efficiently managing API usage, such as implementing batch processing for OCR when dealing with multiple images 4 and introducing delays between API calls 25, can help avoid hitting these limits.
The pricing for Mistral AI's APIs varies depending on the model and the usage 2. As of March 2025, the pricing for the mistral-ocr-latest model is $1 per 1000 pages, with potentially lower costs for batch inference 2. For the Chat Completion API, pricing is based on the number of input and output tokens. For instance, Mistral Small is priced at $0.2 per million input tokens and $0.6 per million output tokens, while Mistral Large costs $2 per million input tokens and $6 per million output tokens 32. Mistral AI also offers a free tier on La Plateforme, allowing developers to experiment with certain models at no cost 32.
Table 1: Mistral AI API Pricing (as of March 2025)




API
Model
Pricing Unit
Cost
OCR
mistral-ocr-latest
Per 1000 pages
$1
Chat Completion
mistral-small-latest
Per Million Input Tokens
$0.2
Chat Completion
mistral-small-latest
Per Million Output Tokens
$0.6
Chat Completion
mistral-large-latest
Per Million Input Tokens
$2
Chat Completion
mistral-large-latest
Per Million Output Tokens
$6

Conclusion
Mistral AI's OCR and Chat Completion APIs provide a powerful and flexible solution for automating the extraction of structured data from images. By combining the accurate text recognition capabilities of the OCR API with the intelligent structuring abilities of the Chat Completion API, guided by a well-defined schema, users can efficiently transform visual information into usable data. Implementing appropriate image preprocessing techniques, robust error handling, and careful management of API usage within the specified rate limits are essential for building reliable and scalable applications for this purpose. The ongoing advancements in Mistral AI's models suggest continued improvements in accuracy and efficiency, further enhancing the potential for leveraging these APIs in various domains.
Works cited
Evaluating Mistral OCR with Label Studio, accessed March 20, 2025, https://labelstud.io/blog/evaluating-mistral-ocr-with-label-studio/
Mistral OCR: A Guide With Practical Examples | DataCamp, accessed March 20, 2025, https://www.datacamp.com/tutorial/mistral-ocr
OCR and Document Understanding | Mistral AI Large Language ..., accessed March 20, 2025, https://docs.mistral.ai/capabilities/document/
Parse and Extract Data from Documents/Images with Mistral OCR | n8n workflow template, accessed March 20, 2025, https://n8n.io/workflows/3102-parse-and-extract-data-from-documentsimages-with-mistral-ocr/
What is Mistral OCR? Introducing the World's Best Document Understanding API - Apidog, accessed March 20, 2025, https://apidog.com/blog/mistral-ocr/
Mistral OCR: The Document Understanding API That's Making My Developer Life 1000% Easier! - Sebastian Petrus, accessed March 20, 2025, https://sebastian-petrus.medium.com/mistral-ocr-6b3dcc084885
Mistral OCR | Mistral AI, accessed March 20, 2025, https://mistral.ai/news/mistral-ocr
Mistral-7B-Instruct | AI/ML API Documentation, accessed March 20, 2025, https://docs.aimlapi.com/api-references/text-models-llm/mistral-ai/mistral-7b-instruct
Quickstart | Mistral AI Large Language Models, accessed March 20, 2025, https://docs.mistral.ai/getting-started/quickstart/
Bienvenue to Mistral AI Documentation | Mistral AI Large Language ..., accessed March 20, 2025, https://docs.mistral.ai/
Mistral AI API | Documentation | Postman API Network, accessed March 20, 2025, https://www.postman.com/ai-engineer/generative-ai-apis/documentation/00mfx1p/mistral-ai-api?entity=folder-7643177-eba06cb7-e867-4930-a992-9f989e2dc359
Mistral AI API - LiteLLM, accessed March 20, 2025, https://docs.litellm.ai/docs/providers/mistral
Clients | Mistral AI Large Language Models, accessed March 20, 2025, https://docs.mistral.ai/getting-started/clients/
Text generation | Mistral AI Large Language Models, accessed March 20, 2025, https://docs.mistral.ai/capabilities/completion/
LLM Extract - Firecrawl, accessed March 20, 2025, https://docs.firecrawl.dev/features/llm-extract
Schemas - LLM, accessed March 20, 2025, https://llm.datasette.io/en/stable/schemas.html
LLMs for Structured Data Extraction from PDF | Comparing Approaches - Unstract, accessed March 20, 2025, https://unstract.com/blog/comparing-approaches-for-using-llms-for-structured-data-extraction-from-pdfs/
Structured data extraction from unstructured content using LLM schemas, accessed March 20, 2025, https://simonwillison.net/2025/Feb/28/llm-schemas/
mistralai/client-python: Python client library for Mistral AI ... - GitHub, accessed March 20, 2025, https://github.com/mistralai/client-python
7 steps of image pre-processing to improve OCR using Python, accessed March 20, 2025, https://nextgeninvent.com/blogs/7-steps-of-image-pre-processing-to-improve-ocr-using-python-2/
www.docuclipper.com, accessed March 20, 2025, https://www.docuclipper.com/blog/ocr-preprocessing/#:~:text=OCR%20preprocessing%20gets%20documents%20ready,data%20more%20accurate%20and%20useful.
OCR Preprocessing: How to Improve Your OCR Data Extraction Outcome - DocuClipper, accessed March 20, 2025, https://www.docuclipper.com/blog/ocr-preprocessing/
Analysis and Benchmarking of OCR Accuracy for Data Extraction Models - Docsumo, accessed March 20, 2025, https://www.docsumo.com/blogs/ocr/accuracy
Expected the last role to be user but received a different role. Ensure proper role assignment in your configuration. - Portkey, accessed March 20, 2025, https://portkey.ai/error-library/role-assignment-error-10145
Codestral API Tutorial: Getting Started With Mistral's API | DataCamp, accessed March 20, 2025, https://www.datacamp.com/tutorial/codestral-api-tutorial
My mistral-large-2407 serverless deployment api is suddenly failing - Microsoft Q&A, accessed March 20, 2025, https://learn.microsoft.com/en-us/answers/questions/2117664/my-mistral-large-2407-serverless-deployment-api-is
How to get your Mistral AI API key (5 steps) - Merge, accessed March 20, 2025, https://www.merge.dev/blog/mistral-ai-api-key
www.merge.dev, accessed March 20, 2025, https://www.merge.dev/blog/mistral-ai-api-key#:~:text=The%20rate%20limits%20are%20also,or%201%2C000%2C000%20tokens%20per%20month.
What are the limits of the Free-tier? | Mistral AI - Help Center, accessed March 20, 2025, https://help.mistral.ai/en/articles/225174-what-are-the-limits-of-the-free-tier
Rate limit and usage tiers | Mistral AI Large Language Models, accessed March 20, 2025, https://docs.mistral.ai/deployment/laplateforme/tier/
Mistral AI Rate limit · langchain-ai langchainjs · Discussion #4408 - GitHub, accessed March 20, 2025, https://github.com/langchain-ai/langchainjs/discussions/4408
AI in abundance | Mistral AI, accessed March 20, 2025, https://mistral.ai/news/september-24-release/?utm_source=tldrai
Mistral AI Solution Overview: Models, Pricing, and API - Acorn Labs, accessed March 20, 2025, https://www.acorn.io/resources/learning-center/mistral-ai/
Mistral AI lowers prices and offers free API for developers - ai-rockstars.com -, accessed March 20, 2025, https://ai-rockstars.com/mistral-ai-small-pixtral-12b/
