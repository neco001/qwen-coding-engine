# WANX API CONSTANTS (DashScope International Spec)

# Strict Size Mapping (width*height)
# 1664*928  (16:9)
# 1472*1104 (4:3)
# 1328*1328 (1:1)
# 1104*1472 (3:4)
# 928*1664  (9:16)
ASPECT_RATIO_MAP = {
    "16:9": "1664*928",
    "4:3": "1472*1104",
    "1:1": "1328*1328",
    "3:4": "1104*1472",
    "9:16": "928*1664"
}

# Model Constraints
MODEL_LIMITS = {
    "qwen-image-max": {
        "max_n": 1,
        "requires_images": False
    },
    "qwen-image-edit-max": {
        "max_n": 6,
        "requires_images": True
    }
}

# Default settings
DEFAULT_MODEL = "qwen-image-max"
DEFAULT_SIZE = "1328*1328" # 1:1
