import pytest
import tempfile
import json
import os
import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from qwen_mcp.registry import ModelRegistry


def test_model_registry_filtering_and_scoring():
    # Test data mimicking raw metadata with excluded keywords
    raw_metadata = [
        {"id": "qwen-audio-1.0", "name": "qwen-audio-1.0"},
        {"id": "qwen-tts-v2", "name": "qwen-tts-v2"},
        {"id": "qwen-video-pro", "name": "qwen-video-pro"},
        {"id": "qwen-omni-beta", "name": "qwen-omni-beta"},
        {"id": "qwen-image", "name": "qwen-image"},
        {"id": "qwen-image-preview-2025", "name": "qwen-image-preview-2025"},
        {"id": "qwen-image-preview-2026", "name": "qwen-image-preview-2026"},
        {"id": "qwen-text", "name": "qwen-text"},
        {"id": "qwen-text-2024", "name": "qwen-text-2024"},
        {"id": "qwen-text-2025", "name": "qwen-text-2025"},
        {"id": "qwen-multimodal", "name": "qwen-multimodal"},
        {"id": "qwen-multimodal-2026", "name": "qwen-multimodal-2026"},
    ]

    registry = ModelRegistry()
    filtered_and_scored = registry._filter_and_score_models(raw_metadata)

    # 1. Filtering: ensure models with 'audio', 'tts', 'video', 'omni' are excluded
    excluded_keywords = ['audio', 'tts', 'video', 'omni']
    for model in filtered_and_scored:
        name = model['id'].lower()
        for kw in excluded_keywords:
            assert kw not in name, f"Model '{model['id']}' should have been filtered out due to keyword '{kw}'"

    # Assert only expected models remain
    expected_ids = {
        'qwen-image',
        'qwen-image-preview-2025',
        'qwen-image-preview-2026',
        'qwen-text',
        'qwen-text-2024',
        'qwen-text-2025',
        'qwen-multimodal',
        'qwen-multimodal-2026',
    }
    actual_ids = {m['id'] for m in filtered_and_scored}
    assert actual_ids == expected_ids

    # 2. Scoring Logic Validation
    image_models = [m for m in filtered_and_scored if m['id'].startswith('qwen-image')]
    text_models = [m for m in filtered_and_scored if m['id'].startswith('qwen-text')]

    # Scoring helper (to avoid repetition)
    scores = {m['id']: m['priority_score'] for m in filtered_and_scored}

    # For 'qwen-image': 'qwen-image' (len 10) > 'qwen-image-preview-2026' (len 23)
    assert scores['qwen-image'] > scores['qwen-image-preview-2025']
    assert scores['qwen-image'] > scores['qwen-image-preview-2026']
    # '2026' > '2025' tie-break
    assert scores['qwen-image-preview-2026'] > scores['qwen-image-preview-2025']

    # Text models: 'qwen-text' > 'qwen-text-2025' > 'qwen-text-2024'
    assert scores['qwen-text'] > scores['qwen-text-2025']
    assert scores['qwen-text-2025'] > scores['qwen-text-2024']


def test_cache_schema_version_and_priority_score():
    registry = ModelRegistry()
    registry.metadata = {
        "qwen-text": {"id": "qwen-text", "name": "qwen-text", "priority_score": 95},
        "qwen-image": {"id": "qwen-image", "name": "qwen-image", "priority_score": 92},
    }

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as f:
        cache_path = f.name

    try:
        registry.save_cache_to_path(cache_path)

        with open(cache_path, 'r') as f:
            cached_data = json.load(f)

        assert cached_data.get('schema_version') == 3
        assert 'metadata' in cached_data
        
        # Load via registry method
        new_registry = ModelRegistry()
        new_registry.load_cache_from_path(cache_path)
        assert len(new_registry.metadata) == 2
        assert new_registry.metadata['qwen-text']['priority_score'] == 95
    finally:
        if os.path.exists(cache_path):
            os.unlink(cache_path)

if __name__ == "__main__":
    try:
        test_model_registry_filtering_and_scoring()
        test_cache_schema_version_and_priority_score()
        print("ALL TESTS PASSED")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test failed: {e}")
        sys.exit(1)
