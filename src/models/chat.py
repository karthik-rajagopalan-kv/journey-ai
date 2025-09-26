from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ChatRequest(BaseModel):
    description: str
    session_id: str
    input_message: str

class ChatMessage(BaseModel):
    type: str  # "user" or "ai" 
    content: str
    timestamp: str = None

class ChatHistoryResponse(BaseModel):
    success: bool
    session_id: str
    messages: List[Dict[str, Any]]
    message_count: int

class SummarizeSessionsRequest(BaseModel):
    session_ids: List[str]