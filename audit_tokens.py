import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath('src'))
from qwen_mcp.tools import generate_audit

async def main():
    load_dotenv()
    
    tools_content = Path("src/qwen_mcp/tools.py").read_text(encoding="utf-8")
    client_content = Path("src/qwen_mcp/client.py").read_text(encoding="utf-8")
    
    context = f"client.py:\n{client_content}\n\ntools.py:\n{tools_content}"
    
    issue = '''Właśnie napisałem rzemieślniczo kod do śledzenia zużycia tokenów z API DashScope (Qwen) i estymacji kosztów USD. 
Kod znajduje się w pliku `client.py` (śledzenie tokenów za pomocą flagi `include_usage=True` i akumulacji w `self.session_prompt_tokens`) oraz w `tools.py` w funkcji `generate_lp_blueprint` (kalkulacja `cost = (p_tok / 1000000.0) * (7.0 if p_tok > 240000 else 0.8) + (c_tok / 1000000.0) * 2.0`).

Proszę o pełny, bezwzględny audyt tego rozwiązania oczami Senior DevOpsa. 
Skup się na:
1. Poprawności wyliczeń matematycznych i odczytywania argumentów z OpenAI compatible API.
2. Wydajności i problemach ze streamowaniem. Zauważ, że `include_usage: True` sprawia, że API wysyła pusty `chunk.choices` z samym oknem `chunk.usage` na samym końcu streama.
3. Bezpieczeństwie i potencjalnych błędach (edge cases).'''
    
    print("Oczekuję na Audyt od Qwena...")
    result = await generate_audit(content=issue, context=context)
    print("\n--- AUDIT QWEN ---\n")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
