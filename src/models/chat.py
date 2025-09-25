from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    image_description: str
    session_id: str
    input_message: str