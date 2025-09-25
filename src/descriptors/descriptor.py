import base64
import os
import tempfile
import time
from typing import Optional, List
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from logging import getLogger

from src.llm.ollama import Ollama
from src.models.base64_request import MediaType

logger = getLogger(__name__)

# Prompts for image and video description
DESCRIBE_IMAGE_PROMPT = "Describe the image in a few sentences. Be descriptive and detailed. Only output the description, no other text."
DESCRIBE_IMAGE_SEQUENCE_PROMPT = "Describe the video sequence in a few sentences. Be descriptive and detailed. Only output the description, no other text."


class Descriptor:
    llm_client = Ollama()

    def __init__(self):
        pass

    def _base64_to_image(self, base64_data: str) -> np.ndarray:
        """
        Convert base64 image data to OpenCV image.
        
        Args:
            base64_data: Base64 encoded image data
            
        Returns:
            OpenCV image array
        """
        try:
            # Remove data URL prefix if present
            if "base64," in base64_data:
                base64_data = base64_data.split("base64,")[1]
            
            # Decode base64 to bytes
            img_bytes = base64.b64decode(base64_data)
            
            # Convert to numpy array
            nparr = np.frombuffer(img_bytes, np.uint8)
            
            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image data")
                
            return img
        except Exception as e:
            raise ValueError(f"Error converting base64 to image: {e}")

    def _base64_to_video_frames(self, base64_data: str) -> List[np.ndarray]:
        """
        Convert base64 video data to list of frames.
        
        Args:
            base64_data: Base64 encoded video data
            
        Returns:
            List of OpenCV image arrays (frames)
        """
        try:
            # Remove data URL prefix if present
            if "base64," in base64_data:
                base64_data = base64_data.split("base64,")[1]
            
            # Decode base64 to bytes
            video_bytes = base64.b64decode(base64_data)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmpfile:
                tmpfile.write(video_bytes)
                tmp_path = tmpfile.name
            
            # Read frames
            video = cv2.VideoCapture(tmp_path)
            frames = []
            
            while video.isOpened():
                success, frame = video.read()
                if not success:
                    break
                frames.append(frame)
            
            video.release()
            os.unlink(tmp_path)  # Clean up temp file
            
            if not frames:
                raise ValueError("No frames could be extracted from video")
                
            return frames
        except Exception as e:
            raise ValueError(f"Error converting base64 to video frames: {e}")

    def describe_base64_media(self, media_type: MediaType, base64_data: str) -> str:
        """
        Describe media from base64 data.
        
        Args:
            media_type: Type of media (image or video)
            base64_data: Base64 encoded media data
            
        Returns:
            String description of the media
        """
        try:
            t0 = time.perf_counter()
            
            if media_type == MediaType.IMAGE:
                # Process image
                img = self._base64_to_image(base64_data)
                height, width = img.shape[:2]
                
                # Resize if needed
                if width >= height and width > 640:
                    height = int(0.5 + height * 640 / width)
                    width = 640
                    resize = (width, height)
                elif height > width and height > 640:
                    width = int(0.5 + width * 640 / height)
                    height = 640
                    resize = (width, height)
                else:
                    resize = None
                    logger.info(f"Image size {(width, height)} OK - no need to resize")

                if resize is not None:
                    img = cv2.resize(img, resize, interpolation=cv2.INTER_LANCZOS4)
                    logger.info(f"Resized to {resize}")

                _, buffer = cv2.imencode(".png", img)
                base64Frame = base64.b64encode(buffer).decode("utf-8")
                self.llm_client.bind(images=[base64Frame])
                
                result = self.llm_client.generate(DESCRIBE_IMAGE_PROMPT)
                
            else:  # MediaType.VIDEO
                # Process video
                frames = self._base64_to_video_frames(base64_data)
                logger.info(f"{len(frames)} frames extracted")
                
                # Convert frames to base64
                base64Frames = []
                for frame in frames[::50]:  # Take every 50th frame
                    _, buffer = cv2.imencode(".jpg", frame)
                    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
                
                self.llm_client.bind(images=base64Frames)
                result = self.llm_client.generate(DESCRIBE_IMAGE_SEQUENCE_PROMPT)
            
            response = result.replace("\n", "")
            
            dt = time.perf_counter() - t0
            logger.info(f"Response in {dt:.3f} s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error describing {media_type} from base64: {e}")
            return ""

    def _download_from_url(self, url: str, suffix: Optional[str] = None) -> str:
        """
        Download content from a URL and save it to a temporary file.
        
        Args:
            url: The URL to download from
            suffix: Optional file suffix (e.g. '.jpg', '.mp4'). If not provided, 
                   will be extracted from the URL.
        
        Returns:
            Path to the temporary file containing the downloaded content
        
        Raises:
            ValueError: If URL is invalid or content cannot be downloaded
        """
        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL format: {url}")

            # Get suffix from URL if not provided
            if not suffix:
                suffix = os.path.splitext(parsed.path)[-1]
                if not suffix:
                    # Default to .jpg for images if no extension found
                    suffix = '.jpg'

            # Download content
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmpfile:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmpfile.write(chunk)
                return tmpfile.name

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to download content from URL: {e}")
        except Exception as e:
            raise ValueError(f"Error processing URL: {e}")
    def describe_image(self, url: str) -> str:
        """
        Describe an image from a URL.
        
        Args:
            url: URL to the image to be described
        
        Returns:
            String description of the image
        """
        try:
            t0 = time.perf_counter()
            
            # Download from URL
            tmp_path = self._download_from_url(url)
            
            # Process the image
            img = cv2.imread(tmp_path)
            if img is None:
                raise ValueError("Failed to read image file")
            height, width, channels = img.shape

            if width >= height and width > 640:
                height = int(0.5 + height * 640 / width)
                width = 640
                resize = (width, height)
            elif height > width and height > 640:
                width = int(0.5 + width * 640 / height)
                height = 640
                resize = (width, height)
            else:
                resize = None
                logger.info(f"Image size {(width, height)} OK - no need to resize")

            if resize is not None:
                img = cv2.resize(img, resize, interpolation=cv2.INTER_LANCZOS4)
                logger.info(f"Resized to {resize}")

            _, buffer = cv2.imencode(".png", img)
            base64Frame = base64.b64encode(buffer).decode("utf-8")
            self.llm_client.bind(images=[base64Frame])

            result = self.llm_client.generate(DESCRIBE_IMAGE_PROMPT)

            response = result.replace("\n", "")

            dt = time.perf_counter() - t0

            logger.info(f"Response in {dt:.3f} s")
            return response
        except Exception as e:
            logger.error(f"Error describing image from URL: {e}")
            return ""

    def describe_video(self, url: str) -> str:
        """
        Describe a video from a URL.
        
        Args:
            url: URL to the video to be described
        
        Returns:
            String description of the video
        """
        try:
            t0 = time.perf_counter()
            
            # Download from URL
            tmp_path = self._download_from_url(url)

            # Process the video
            video = cv2.VideoCapture(tmp_path)

            base64Frames = []
            while video.isOpened():
                success, frame = video.read()
                if not success:
                    break
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

            video.release()
            logger.info(f"{len(base64Frames)} frames read.")

            self.llm_client.bind(images=base64Frames[0::50])

            result = self.llm_client.generate(DESCRIBE_IMAGE_SEQUENCE_PROMPT)
            response = result.replace("\n", "")

            dt = time.perf_counter() - t0
            logger.info(f"Respone in {dt:.3f} s")

            return response
        except Exception as e:
            logger.error(f"Error describing video from URL: {e}")
            return ""
