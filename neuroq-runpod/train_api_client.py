#!/usr/bin/env python3
"""
NeuroQ Train API Client

Modal FastAPI エンドポイントを使用してトレーニングを実行するクライアント。

使用例:
    # デフォルト（micro モデル, oasst1-ja, early_stop）
    python train_api_client.py

    # モデルサイズとデータセットを指定
    python train_api_client.py --model-size small --datasets kunishou/oasst1-89k-ja OpenAssistant/oasst1

    # エポック数とサンプル数を指定
    python train_api_client.py --epochs 20 --max-samples 5000 --epoch-mode fixed

    # config 付きデータセットを指定
    python train_api_client.py --datasets-with-config '[{"id":"tokyotech-llm/swallow-code-v2","config":"stage1-auto-format"}]'
"""

import argparse
import json
import sys
import time

import requests

BASE_URL = "https://higuchiyuya-riddle--neuroq-api-fastapi-app.modal.run"

# Modal API は 303 リダイレクトで結果を返すため -L (follow redirects) 相当が必要
SESSION = requests.Session()
SESSION.max_redirects = 20


def health_check() -> dict:
    """ヘルスチェック"""
    resp = SESSION.get(f"{BASE_URL}/health", timeout=300, allow_redirects=True)
    resp.raise_for_status()
    return resp.json()


def status_check() -> dict:
    """ステータス確認"""
    resp = SESSION.get(f"{BASE_URL}/status", timeout=300, allow_redirects=True)
    resp.raise_for_status()
    return resp.json()


def start_training(
    model_size: str = "micro",
    epochs: int = 10,
    batch_size: int = None,
    learning_rate: float = None,
    dataset_ids: list = None,
    datasets_with_config: list = None,
    text_column: str = "text",
    split: str = "train",
    max_samples: int = None,
    epoch_mode: str = "early_stop",
    hf_token: str = None,
) -> dict:
    """トレーニング開始"""
    payload = {
        "model_size": model_size,
        "epochs": epochs,
        "text_column": text_column,
        "split": split,
        "epoch_mode": epoch_mode,
    }
    if batch_size is not None:
        payload["batch_size"] = batch_size
    if learning_rate is not None:
        payload["learning_rate"] = learning_rate
    if dataset_ids:
        payload["dataset_ids"] = dataset_ids
    if datasets_with_config:
        payload["datasets"] = datasets_with_config
    if max_samples is not None:
        payload["max_samples"] = max_samples
    if hf_token:
        payload["hf_token"] = hf_token

    resp = SESSION.post(
        f"{BASE_URL}/train",
        json=payload,
        timeout=300,
        allow_redirects=True,
    )
    resp.raise_for_status()
    return resp.json()


def poll_training_status(call_id: str, interval: int = 30) -> dict:
    """トレーニング完了までポーリング"""
    print(f"\nTraining call_id: {call_id}")
    print(f"Polling every {interval}s ...")

    elapsed = 0
    while True:
        time.sleep(interval)
        elapsed += interval

        try:
            resp = SESSION.get(
                f"{BASE_URL}/train/status/{call_id}",
                timeout=300,
                allow_redirects=True,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [{elapsed}s] Status check failed: {e}")
            continue

        status = data.get("status", "unknown")
        if status == "completed":
            print(f"  [{elapsed}s] Training completed!")
            return data
        elif status == "running":
            print(f"  [{elapsed}s] Still running...")
        else:
            print(f"  [{elapsed}s] Status: {status} - {data}")
            if status in ("failed", "error"):
                return data

    return data


def generate_text(prompt: str, model_size: str = "micro", max_length: int = 50) -> dict:
    """テキスト生成"""
    resp = SESSION.post(
        f"{BASE_URL}/generate",
        json={
            "prompt": prompt,
            "model_size": model_size,
            "max_length": max_length,
            "temperature": 0.7,
        },
        timeout=300,
        allow_redirects=True,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="NeuroQ Train API Client")
    sub = parser.add_subparsers(dest="command", help="コマンド")

    # train
    train_p = sub.add_parser("train", help="トレーニング実行")
    train_p.add_argument("--model-size", default="micro", choices=["micro", "small", "large"])
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.add_argument("--batch-size", type=int, default=None)
    train_p.add_argument("--learning-rate", type=float, default=None)
    train_p.add_argument("--datasets", nargs="+", default=None, help="HuggingFace dataset IDs")
    train_p.add_argument(
        "--datasets-with-config",
        type=str,
        default=None,
        help='JSON: [{"id":"repo/name","config":"subset"}]',
    )
    train_p.add_argument("--text-column", default="text")
    train_p.add_argument("--split", default="train")
    train_p.add_argument("--max-samples", type=int, default=None)
    train_p.add_argument("--epoch-mode", default="early_stop", choices=["early_stop", "fixed"])
    train_p.add_argument("--hf-token", default=None)
    train_p.add_argument("--poll-interval", type=int, default=30, help="ポーリング間隔（秒）")
    train_p.add_argument("--no-wait", action="store_true", help="開始のみ（完了を待たない）")

    # status
    status_p = sub.add_parser("status", help="API ステータス確認")

    # health
    health_p = sub.add_parser("health", help="ヘルスチェック")

    # train-status
    ts_p = sub.add_parser("train-status", help="トレーニングジョブのステータス確認")
    ts_p.add_argument("call_id", help="train API が返した call_id")

    # generate
    gen_p = sub.add_parser("generate", help="テキスト生成")
    gen_p.add_argument("prompt", help="生成プロンプト")
    gen_p.add_argument("--model-size", default="micro")
    gen_p.add_argument("--max-length", type=int, default=50)

    args = parser.parse_args()

    if args.command is None:
        # デフォルト: micro モデルで oasst1-ja トレーニング
        args.command = "train"
        args.model_size = "micro"
        args.epochs = 10
        args.batch_size = None
        args.learning_rate = None
        args.datasets = ["kunishou/oasst1-89k-ja"]
        args.datasets_with_config = None
        args.text_column = "text"
        args.split = "train"
        args.max_samples = 1000
        args.epoch_mode = "early_stop"
        args.hf_token = None
        args.poll_interval = 30
        args.no_wait = False

    if args.command == "health":
        data = health_check()
        print(json.dumps(data, indent=2, ensure_ascii=False))

    elif args.command == "status":
        data = status_check()
        print(json.dumps(data, indent=2, ensure_ascii=False))

    elif args.command == "train-status":
        resp = SESSION.get(
            f"{BASE_URL}/train/status/{args.call_id}",
            timeout=300,
            allow_redirects=True,
        )
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

    elif args.command == "generate":
        data = generate_text(args.prompt, args.model_size, args.max_length)
        print(json.dumps(data, indent=2, ensure_ascii=False))

    elif args.command == "train":
        # 1. ヘルスチェック
        print("=== NeuroQ Train API Client ===\n")
        print("Checking API health...")
        try:
            h = health_check()
            print(f"  Health: {h.get('status')} | CUDA: {h.get('cuda_available')} | Device: {h.get('device')}")
        except Exception as e:
            print(f"  Health check failed: {e}")
            print("  Proceeding anyway...\n")

        # 2. トレーニング開始
        datasets_with_config = None
        if args.datasets_with_config:
            datasets_with_config = json.loads(args.datasets_with_config)

        print(f"\nStarting training:")
        print(f"  Model: {args.model_size}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Mode: {args.epoch_mode}")
        print(f"  Datasets: {args.datasets or datasets_with_config or 'default'}")
        if args.max_samples:
            print(f"  Max samples: {args.max_samples}")
        print()

        result = start_training(
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dataset_ids=args.datasets,
            datasets_with_config=datasets_with_config,
            text_column=args.text_column,
            split=args.split,
            max_samples=args.max_samples,
            epoch_mode=args.epoch_mode,
            hf_token=args.hf_token,
        )
        print(f"Train response: {json.dumps(result, indent=2, ensure_ascii=False)}")

        call_id = result.get("call_id")
        if not call_id:
            print("No call_id returned. Training may have failed.")
            sys.exit(1)

        if args.no_wait:
            print(f"\ncall_id: {call_id}")
            print("Use 'train-status' command to check later.")
            sys.exit(0)

        # 3. 完了までポーリング
        final = poll_training_status(call_id, interval=args.poll_interval)
        print(f"\nFinal result: {json.dumps(final, indent=2, ensure_ascii=False)}")

        # 4. トレーニング後にテスト生成
        if final.get("status") == "completed":
            print("\n=== Post-training test generation ===")
            for prompt in ["こんにちは", "AI とは", "Hello"]:
                try:
                    gen = generate_text(prompt, args.model_size, 50)
                    print(f"  Q: {prompt}")
                    print(f"  A: {gen.get('generated', '(no output)')[:200]}")
                    print()
                except Exception as e:
                    print(f"  Generation failed for '{prompt}': {e}\n")


if __name__ == "__main__":
    main()
