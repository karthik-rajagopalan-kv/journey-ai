from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"

class Base64Request(BaseModel):
    media_type: MediaType = Field(..., description="Type of media (image or video)")
    base64_data: str = Field(..., description="Base64 encoded media data")
    content_type: Optional[str] = Field(None, description="Content type of the media (e.g. image/jpeg, video/mp4)")
