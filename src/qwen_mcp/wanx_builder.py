import base64
import os
import mimetypes
from typing import List, Dict, Any, Optional

class WanxPayloadBuilder:
    def __init__(self, model: str = "qwen-image-edit-max", size: str = "1024*1024"):
        self.model = model
        self.prompt = ""
        self.images = []
        self.size = size
        self.n = 1
        self.negative_prompt = "low quality, bad proportions, blurry, text, digits"
        self.prompt_extend = True

    def set_model(self, model: str) -> 'WanxPayloadBuilder':
        self.model = model
        return self

    def set_prompt(self, prompt: str) -> 'WanxPayloadBuilder':
        self.prompt = prompt
        return self

    def set_images(self, images: List[str]) -> 'WanxPayloadBuilder':
        if len(images) > 3:
            # Instead of raising we can just slice to be more permissive with high-level tools
            self.images = images[:3]
        else:
            self.images = images
        return self

    def set_size(self, size: str) -> 'WanxPayloadBuilder':
        self.size = size
        return self

    def set_n(self, n: int) -> 'WanxPayloadBuilder':
        self.n = n
        return self

    def set_negative_prompt(self, negative_prompt: str) -> 'WanxPayloadBuilder':
        self.negative_prompt = negative_prompt
        return self

    def set_prompt_extend(self, prompt_extend: bool) -> 'WanxPayloadBuilder':
        self.prompt_extend = prompt_extend
        return self

    def build(self) -> Dict[str, Any]:
        """Constructs the final DashScope WanX 2.1 payload."""
        content = []
        
        # 1. Images (1 to 3 objects)
        for i, path in enumerate(self.images[:3]):
            if not path:
                continue
                
            if path.startswith(("http://", "https://", "data:image")):
                content.append({"image": path})
            elif os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                        b64_data = base64.b64encode(data).decode("utf-8")
                    # Try to guess mime, fallback to jpeg
                    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
                    content.append({"image": f"data:{mime};base64,{b64_data}"})
                except Exception as e:
                    # In dry_run we maybe want to see the error, but builder should be robust
                    content.append({"image": f"ERROR: Failed to encode {path}: {str(e)}"})
            else:
                # If it's not a URL and not a local file, it might be an orphaned string
                # We append it as-is (maybe it's a temp URL from another tool)
                content.append({"image": path})

        # 2. Text instruction (Exactly one)
        content.append({"text": self.prompt or "Generate an image based on references."})

        return {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            },
            "parameters": {
                "n": self.n,
                "negative_prompt": self.negative_prompt,
                "prompt_extend": self.prompt_extend,
                "watermark": False,
                "size": self.size
            }
        }
