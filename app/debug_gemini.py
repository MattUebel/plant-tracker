import sys
import os
import traceback

try:
    print("Python path:", sys.path)
    print("Current directory:", os.getcwd())
    print("Files in utils:", os.listdir("utils"))
    
    try:
        import google.generativeai as genai
        print("Successfully imported google.generativeai")
    except ImportError as e:
        print(f"Error importing google.generativeai: {e}")

    try:
        from utils.gemini_vision_api import GeminiVisionTester
        print("Successfully imported GeminiVisionTester")
    except ImportError as e:
        print(f"Error importing GeminiVisionTester: {e}")
        traceback.print_exc()
        
except Exception as e:
    print(f"General error: {e}")
    traceback.print_exc()

