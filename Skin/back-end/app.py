from flask import Flask, request, jsonify
import base64
import requests
import os
import logging
from flask_cors import CORS
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    logger.info("GEMINI_API_KEY loaded successfully")
else:
    logger.error("GEMINI_API_KEY not found! Check your .env file")

GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# Basic skin analysis function with improved prompt for categorization
def analyze_skin_with_categories(image_bytes):
    try:
        base64_img = base64.b64encode(image_bytes).decode("utf-8")
        
        # Enhanced prompt that requests structured information including skin type categorization
        prompt = """Analyze this person's skin condition and provide a structured analysis with the following sections:

Skin Type: [Determine if the skin is Oily, Dry, Combination, Normal, or Sensitive]

Hydration Level: [Rate as Low, Moderate, or High]

Key Conditions: [List any visible acne, redness, dryness, etc.]

Recommendations: [Suggest 2-3 short skincare advice points]

Format the response with each section clearly labeled. Use the format "Skin Type: [type]" etc.
"""
        
        body = {
            "contents": [{
                "parts": [
                    {
                        "text": prompt
                    },
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64_img
                        }
                    }
                ]
            }]
        }
        headers = {
            "Content-Type": "application/json"
        }
        
        logger.debug(f"Sending request to Gemini API")
        response = requests.post(GEMINI_ENDPOINT, json=body, headers=headers)
        
        if response.status_code == 200:
            logger.debug("Gemini API request successful")
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            logger.error(f"Gemini API error: {response.status_code} - {response.text}")
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        logger.error(f"Exception in analyze_skin_with_categories: {str(e)}")
        return f"Error processing request: {str(e)}"

# Original endpoint that frontend calls - Enhanced with categorization
@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
        
    try:
        logger.info("Received request to /analyze endpoint")
        data = request.get_json()
        if not data or "image" not in data:
            logger.error("No image data in request")
            return jsonify({"error": "No image data provided"}), 400
            
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        
        # Use the enhanced analysis function that includes categorization
        result = analyze_skin_with_categories(image_bytes)
        
        return jsonify({"result": result})
    except Exception as e:
        logger.error(f"Error in /analyze: {str(e)}")
        return jsonify({"error": str(e)}), 500

# More detailed skin analysis for future expansion
def detailed_skin_analysis(image_bytes):
    prompt = """Provide a comprehensive skin analysis with the following details:

Skin Type: [Oily/Dry/Combination/Normal/Sensitive]
Hydration Level: [Low/Moderate/High]
Skin Texture: [Smooth/Rough/Uneven]
Visible Concerns:
- Acne: [None/Mild/Moderate/Severe]
- Hyperpigmentation: [None/Mild/Moderate/Severe]
- Fine Lines: [None/Mild/Moderate/Severe] 
- Redness: [None/Mild/Moderate/Severe]
- Enlarged Pores: [None/Mild/Moderate/Severe]
- Sun Damage: [None/Mild/Moderate/Severe]

Top 3 Recommendations:
1. [First recommendation]
2. [Second recommendation]
3. [Third recommendation]

Format the response with clear labels for each section.
"""
    return query_gemini(image_bytes, prompt)

# CORS helper function
def _build_cors_preflight_response():
    response = jsonify({})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
    return response

# Generic function to query Gemini with different prompts
def query_gemini(image_bytes, prompt):
    try:
        base64_img = base64.b64encode(image_bytes).decode("utf-8")
        body = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/jpeg", "data": base64_img}}
                ]
            }]
        }
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(GEMINI_ENDPOINT, json=body, headers=headers)
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error processing request: {str(e)}"

# Additional endpoint for more detailed analysis (for future use)
@app.route("/analyze/detailed", methods=["POST", "OPTIONS"])
def analyze_detailed():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
        
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400
            
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        result = detailed_skin_analysis(image_bytes)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Simple route to test if server is up
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "message": "Server is running"})

if __name__ == "__main__":
    logger.info("Starting server on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0')