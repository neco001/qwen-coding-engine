import os
import requests
import base64
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Wczytujemy zmienne środowiskowe
load_dotenv()

def get_token_usage():
    """
    Pobiera statystyki zużycia tokenów z Alibaba Cloud Prometheus API.
    """
    # 1. Konfiguracja (Dane z instrukcji i .env)
    # UWAGA: AccessKey i AccessKeySecret muszą pochodzić z konta Alibaba Cloud (RAM), 
    # nie z DashScope API Key.
    access_key = os.getenv("ALIBABA_ACCESS_KEY")
    access_secret = os.getenv("ALIBABA_ACCESS_SECRET")
    
    # Endpoint Prometheus HTTP API (z konsoli Alibaba Model Monitoring)
    # Domyślnie używamy Singapore, chyba że wolisz inny.
    prometheus_endpoint = os.getenv("ALIBABA_PROMETHEUS_ENDPOINT", "https://coding-intl.dashscope.aliyuncs.com")
    
    if not access_key or not access_secret:
        print("❌ BŁĄD: Brakuje ALIBABA_ACCESS_KEY lub ALIBABA_ACCESS_SECRET w pliku .env")
        print("Są one wymagane do autoryzacji w Prometheus API (to inne klucze niż DashScope API Key).")
        return

    # 2. Przygotowanie autoryzacji (Basic Auth base64)
    auth_str = f"{access_key}:{access_secret}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Basic {encoded_auth}"
    }

    # 3. Parametry czasu (Ostatnie 24h)
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)
    
    # Formatowanie dla API: 2025-11-20T00:00:00Z
    params = {
        "query": "model_usage",  # Podstawowy metryka zużycia
        "start": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "step": "1h"  # Krok godzinny
    }

    print(f"📡 Pobieram dane z: {prometheus_endpoint}")
    print(f"📅 Zakres: {params['start']} do {params['end']}")

    try:
        url = f"{prometheus_endpoint}/api/v1/query_range"
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                results = data['data']['result']
                if not results:
                    print("📭 Brak danych o zużyciu w podanym okresie.")
                    return
                
                print("\n📊 RAPORT ZUŻYCIA TOKENÓW (Ostatnie 24h):")
                print("-" * 50)
                for res in results:
                    model = res['metric'].get('model', 'unknown')
                    workspace = res['metric'].get('workspace_id', 'unknown')
                    values = res['values']
                    
                    total_in_period = sum(float(val[1]) for val in values)
                    print(f"🔹 Model: {model}")
                    print(f"   Workspace: {workspace}")
                    print(f"   Suma tokenów (szacunkowa): {total_in_period:.0f}")
                    print("-" * 50)
            else:
                print(f"❌ Błąd API: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Błąd połączenia ({response.status_code}): {response.text}")

    except Exception as e:
        print(f"❌ Wystąpił błąd: {e}")

if __name__ == "__main__":
    get_token_usage()
