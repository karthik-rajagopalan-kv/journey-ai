from pydantic import BaseModel

class ImageRequest(BaseModel):
    session_id: str
    model: str = "seedream-4-0-250828"
    sequential_image_generation: str = "disabled"
    response_format: str = "url"
    size: str = "2K"
    stream: bool = False
    watermark: bool = False
