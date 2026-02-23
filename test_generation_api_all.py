import requests
import json
import time

API_URL = "https://higuchiyuya-riddle--neuroq-api-fastapi-app.modal.run/generate"
MODELS = ["micro", "small", "large"]

PROMPTS = [
    "User: こんにちは。自己紹介をしてください。\nAssistant:",
    "User: Pythonでフィボナッチ数列を生成する関数を書いてください。\nAssistant:",
    "User: 12 + 15 * 3 - 4 = ? 計算手順と一緒に教えてください。\nAssistant:"
]

def test_generation():
    print("=" * 70)
    print("🧠 NeuroQ API Generation Test")
    print("=" * 70)

    for model_size in MODELS:
        print(f"\n🚀 Testing Model: {model_size.upper()}")
        print("-" * 50)
        
        for prompt in PROMPTS:
            payload = {
                "prompt": prompt,
                "model_size": model_size,
                "max_length": 64, # Keep short for quick testing
                "temperature": 0.7,
            }
            
            try:
                start_time = time.time()
                response = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"})
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    generated_text = data.get("generated_text", "")
                    
                    print(f"\n[Prompt]: {prompt.strip()}")
                    print(f"[Output]: {generated_text}")
                    print(f"[Time]: {elapsed:.2f}s")
                else:
                    print(f"\n❌ Error ({response.status_code}): {response.text}")
                    
            except Exception as e:
                print(f"❌ Connection error: {e}")
                
            time.sleep(1) # Polite delay

if __name__ == "__main__":
    test_generation()
