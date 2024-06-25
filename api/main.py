# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from gramformer import Gramformer
import concurrent.futures
import torch
import httpx
from dotenv import load_dotenv
import os
import logging
import spacy

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a request model
class TextRequest(BaseModel):
    text: str

# API details from environment variables
TRINKA_API_URL = "https://api-platform.trinka.ai/api/v2/plugin/check/paragraph"
TRINKA_API_KEY = os.getenv('TRINKA_API_KEY')
if not TRINKA_API_KEY:
    raise ValueError("TRINKA_API_KEY is not set in the environment variables")

TEXTGEARS_API_URL = "https://api.textgears.com/grammar"
TEXTGEARS_API_KEY = os.getenv('TEXTGEARS_API_KEY')
if not TEXTGEARS_API_KEY:
    raise ValueError("TEXTGEARS_API_KEY is not set in the environment variables")

LANGUAGETOOL_API_URL = "https://api.languagetool.org/v2/check"

# Ensure SpaCy model is installed
def ensure_spacy_model(model_name="en_core_web_sm"):
    try:
        spacy.load(model_name)
    except OSError:
        print(f"Downloading SpaCy model '{model_name}'")
        os.system(f"pipenv run python -m spacy download {model_name}")

ensure_spacy_model()

# Setup Gramformer
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(1212)
gf = Gramformer(models=1, use_gpu=False)  # 1=corrector, 2=detector

async def call_trinka_api(client, text):
    data_trinka = {
        "paragraph": text,
        "language": "US",
        "style_guide": "",
        "is_sensitive_data": False,
    }
    headers_trinka = {
        "x-api-key": TRINKA_API_KEY,
        "Content-Type": "application/json",
    }
    response = await client.post(TRINKA_API_URL, json=data_trinka, headers=headers_trinka)
    response.raise_for_status()
    return response.json()

async def call_textgears_api(client, text):
    params_textgears = {
        "key": TEXTGEARS_API_KEY,
        "text": text,
    }
    response = await client.post(TEXTGEARS_API_URL, params=params_textgears)
    response.raise_for_status()
    return response.json()

async def call_languagetool_api(client, text):
    params_languagetool = {
        "text": text,
        "language": "en-US",
    }
    response = await client.post(LANGUAGETOOL_API_URL, params=params_languagetool)
    response.raise_for_status()
    return response.json()

def correct_text_with_gramformer(text):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(gf.correct, text, max_candidates=1)
        corrected_sentences = future.result()
    results = []
    for corrected_sentence in corrected_sentences:
        edits = gf.get_edits(text, corrected_sentence)
        results.append({
            "corrected": corrected_sentence,
            "edits": edits
        })
    return results

@app.post("/api/v1/check-text")
async def receive_text(request: TextRequest):
    logger.info("Received text: %s", request.text)
    async with httpx.AsyncClient() as client:
        try:
            trinka_data = await call_trinka_api(client, request.text)
            textgears_data = await call_textgears_api(client, request.text)
            languagetool_data = await call_languagetool_api(client, request.text)
            gramformer_data = correct_text_with_gramformer(request.text)
        except httpx.RequestError as exc:
            logger.error("Request error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Request error: {exc}")
        except httpx.HTTPStatusError as exc:
            logger.error("HTTP error: %s", exc)
            raise HTTPException(status_code=exc.response.status_code, detail=f"HTTP error: {exc}")

    combined_data = {
        "trinka": trinka_data,
        "textgears": textgears_data,
        "languagetool": languagetool_data,
        "gramformer": gramformer_data
    }
    
    return JSONResponse(content=combined_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
