import urllib.request
import json
import time

API_URL = "https://higuchiyuya-riddle--neuroq-api-fastapi-app.modal.run/train"
STATUS_URL = "https://higuchiyuya-riddle--neuroq-api-fastapi-app.modal.run/train/status"

DATASETS = [
    {"id": "kunishou/oasst1-89k-ja"},
    {"id": "OpenAssistant/oasst2"}
]

models = ["micro", "small", "large"]


def start_training():
    call_ids = []
    for model in models:
        print(f"🚀 Starting OASST1 & OASST2 training for {model}...")
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
            with urllib.request.urlopen(req, timeout=120) as response:
                if response.status == 200:
                    response_data = json.loads(response.read().decode('utf-8'))
                    call_id = response_data.get('call_id')
                    call_ids.append((model, call_id))
                    print(f"✅ Success! Call ID: {call_id}")
                    print(f"   Response: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
                else:
                    print(f"❌ Failed: {response.status}")
        except Exception as e:
            print(f"❌ Error: {e}")

        time.sleep(2)

    print("\n📋 Training Jobs Summary:")
    for model, call_id in call_ids:
        print(f"  {model}: {call_id}")
        if call_id:
            print(f"    Status URL: {STATUS_URL}/{call_id}")

    return call_ids


if __name__ == "__main__":
    start_training()
