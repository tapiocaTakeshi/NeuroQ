import urllib.request
import json
import time

API_URL = "https://higuchiyuya-riddle--neuroq-api-fastapi-app.modal.run/train"

DATASETS = [
    {"id": "iamtarun/python_code_instructions_18k_alpaca"},
    {"id": "TIGER-Lab/MathInstruct"}
]

models = ["micro", "small", "large"]

def start_training():
    for model in models:
        print(f"🚀 Starting Code & Math training for {model}...")
        payload = {
            "model_size": model,
            "epochs": 10, 
            "datasets": DATASETS,
            "max_samples": 20000, 
            "epoch_mode": "early_stop"
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(API_URL, data=data, headers={"Content-Type": "application/json"})
        
        try:
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    response_data = json.loads(response.read().decode('utf-8'))
                    print(f"✅ Success! Call ID: {response_data.get('call_id')}")
                else:
                    print(f"❌ Failed: {response.status}")
        except Exception as e:
            print(f"❌ Error: {e}")
            
        time.sleep(2)

if __name__ == "__main__":
    start_training()
