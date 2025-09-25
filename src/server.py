import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from src.chat.chat_service import ChatService
from src.models.chat import ChatRequest
from src.models.image import ImageRequest
from src.models.base64_request import Base64Request
from src.descriptors.descriptor import Descriptor

load_dotenv()

app = FastAPI()

@app.get("/health")
async def health():
    return JSONResponse(content={"status": "healthy"}, status_code=200)

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


@app.post("/journal/chat")
async def chat(request: ChatRequest):
    chat_service = ChatService(request.session_id)
    response = chat_service.chat(request.image_description, request.input_message)
    if response["type"] == "error":
        raise HTTPException(status_code=500, detail=response["content"])
    return JSONResponse(content=response, status_code=200)

@app.post("/descriptions/base64")
async def describe_base64_media(request: Base64Request):
    """
    Describe media content from base64 data.
    
    Args:
        request: Base64Request containing media type and base64 encoded data
        
    Returns:
        JSON response with media description
    """
    try:
        descriptor = Descriptor()
        description = descriptor.describe_base64_media(request.media_type, request.base64_data)
        return JSONResponse(
            content={"description": description},
            status_code=200
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)