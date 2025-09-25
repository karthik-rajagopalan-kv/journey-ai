from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    description: str
    session_id: str
    input_message: str