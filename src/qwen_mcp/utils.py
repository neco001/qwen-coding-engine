
import json
import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Utility to pull JSON blocks from LLM markdown responses."""
    try:
        # Try finding a ```json block first
        match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        json_str = match.group(1) if match else text
        return json.loads(json_str.strip())
    except Exception:
        # Fallback to finding anything that looks like a JSON object
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
    return None
