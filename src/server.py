import os
import random
import requests
import redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from src.chat.agent import Summarizer
from src.chat.chat_service import ChatService
from src.models.chat import ChatRequest, ChatHistoryResponse, SummarizeSessionsRequest
from src.models.image import ImageRequest
from src.models.base64_request import Base64Request
from src.descriptors.descriptor import Descriptor

load_dotenv()

app = FastAPI()
redis_client = redis.Redis(host="localhost", port=6379, db=0)

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

    seed = random.randint(0, 1000000)

    prompt = f"You are a whimsical doodle artist. Create a simple, playful, and imaginative doodle based on the following idea. The style should be minimalist, with clean lines and a hand-drawn feel. Think of a quick sketch in a notebook, but with a touch of digital polish. {request.description}"

    data = {
        "model": request.model,
        "prompt": prompt,
        "seed": seed,
        "sequential_image_generation": request.sequential_image_generation,
        "response_format": request.response_format,
        "size": request.size,
        "stream": request.stream,
        "watermark": request.watermark,
    }

    try:
        response = requests.post(
            "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations", headers=headers, json=data
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        print("Response:", response.json())
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/journal/chat")
async def chat(request: ChatRequest):
    chat_service = ChatService(request.session_id)
    response = chat_service.chat(request.description, request.input_message)
    if response["type"] == "error":
        raise HTTPException(status_code=500, detail=response["content"])
    return JSONResponse(content=response, status_code=200)


@app.get("/chats/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """
    Retrieve chat history for a specific session ID from Redis.

    Args:
        session_id: The session ID to retrieve chat history for

    Returns:
        ChatHistoryResponse containing all messages for the session
    """
    try:
        # Validate session_id
        if not session_id or not session_id.strip():
            raise HTTPException(status_code=400, detail="Session ID cannot be empty")

        # Create ChatService instance to access Redis
        chat_service = ChatService(session_id.strip())

        # Get chat history from Redis
        messages = chat_service.get_chat_history()

        return ChatHistoryResponse(success=True, session_id=session_id, messages=messages, message_count=len(messages))

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle any other unexpected errors
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")


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
        return JSONResponse(content={"description": description}, status_code=200)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/journal/summarize/sessions")
async def summarize_sessions(request: SummarizeSessionsRequest):
    summarizer = Summarizer()
    response = summarizer.summarize_sessions(request.session_ids)
    return JSONResponse(content={"summary": response}, status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)