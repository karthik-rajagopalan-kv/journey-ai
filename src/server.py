import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.models.image import ImageRequest

load_dotenv()

app = FastAPI()

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ARK_API_KEY not set in environment variables")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": request.model,
        "prompt": request.prompt,
        "sequential_image_generation": request.sequential_image_generation,
        "response_format": request.response_format,
        "size": request.size,
        "stream": request.stream,
        "watermark": request.watermark,
    }

    try:
        response = requests.post("https://ark.ap-southeast.bytepluses.com/api/v3/images/generations", headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        print("Response:", response.json())
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
