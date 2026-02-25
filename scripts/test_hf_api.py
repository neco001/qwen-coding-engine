import urllib.request
import json
import ssl


def test_hf_api():
    url = "https://huggingface.co/api/models?author=Qwen&limit=5&full=true"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as response:
            if response.status == 200:
                models = json.loads(response.read().decode())
                for m in models:
                    print(f"Model: {m.get('id')}")
                    print(f"  Pipeline: {m.get('pipeline_tag')}")
                    print(f"  Tags: {m.get('tags')}")
                    # siblings can contain filenames (check for config.json to get param count if needed)
                    # but usually parameter count is in tags like 'region:us', 'license:apache-2.0'
                    # Actually parameter count is often in the 'config' or tags
                    # Let's see if there's a 'safetensors' metadata
                    st = m.get("safetensors")
                    if st:
                        print(f"  Parameters: {st.get('parameters')}")
            else:
                print(f"Error: {response.status}")
    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    test_hf_api()
