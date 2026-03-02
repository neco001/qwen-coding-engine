import base64
import os
import mimetypes
import logging
from typing import List, Dict, Any, Optional
from qwen_mcp.wanx_constants import ASPECT_RATIO_MAP, MODEL_LIMITS, DEFAULT_MODEL, DEFAULT_SIZE

logger = logging.getLogger(__name__)

class WanxPayloadBuilder:
    def __init__(self, model: Optional[str] = None, size: Optional[str] = None):
        self.model = model
        self.size = size
        self.prompt = ""
        self.images = []
        self.n = 1
        self.negative_prompt = "low resolution, low quality, deformed limbs, deformed fingers, oversaturated, waxy, no facial details, overly smooth, AI-like, chaotic composition, blurry text, distorted text."
        self.prompt_extend = True

    def set_model(self, model: str) -> 'WanxPayloadBuilder':
        self.model = model
        return self

    def set_prompt(self, prompt: str) -> 'WanxPayloadBuilder':
        self.prompt = prompt
        return self

    def set_images(self, images: List[str]) -> 'WanxPayloadBuilder':
        # Limit to 3 images as per API spec for edit-max
        self.images = images[:3]
        return self

    def set_size(self, size: str) -> 'WanxPayloadBuilder':
        # size can be "1:1" or "1328*1328"
        self.size = size
        return self

    def set_n(self, n: int) -> 'WanxPayloadBuilder':
        self.n = n
        return self

    def set_prompt_extend(self, prompt_extend: bool) -> 'WanxPayloadBuilder':
        self.prompt_extend = prompt_extend
        return self

    def _encode_image(self, path: str) -> str:
        """Helper to encode local file to base64 with data URI."""
        if path.startswith(("http://", "https://", "data:image")):
            return path
        
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read()
                b64_data = base64.b64encode(data).decode("utf-8")
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            return f"data:{mime};base64,{b64_data}"
        
        return path

    def build(self) -> Dict[str, Any]:
        """Constructs the final DashScope WanX payload."""
        
        # 1. Determine Model
        model_name = self.model
        if not model_name:
            model_name = "qwen-image-edit-max" if self.images else "qwen-image-max"
        
        # 2. Determine Size String
        # Map human-readable (16:9) to API format (1664*928)
        size_str = ASPECT_RATIO_MAP.get(self.size, self.size) if self.size else DEFAULT_SIZE
        if "*" not in size_str: # Final fail-safe
            size_str = DEFAULT_SIZE

        # 3. Model Constraints (Enforce n=1 for qwen-image-max)
        n_val = self.n
        if model_name == "qwen-image-max":
            n_val = 1
        
        # 4. Construct Content Array
        content = []
        
        # Images first
        for img_path in self.images:
            encoded = self._encode_image(img_path)
            content.append({"image": encoded})
        
        # Exactly one text object (Required)
        content.append({"text": self.prompt or "Generate an image."})

        # Final Payload Structure
        return {
            "model": model_name,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            },
            "parameters": {
                "size": size_str,
                "n": n_val,
                "negative_prompt": self.negative_prompt,
                "prompt_extend": self.prompt_extend,
                "watermark": False
            }
        }
