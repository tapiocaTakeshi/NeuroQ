#!/usr/bin/env python3
"""
推奨の日本語データセットでNeuroQを学習するスクリプト
Stage 1: 事前学習 → Stage 2: 会話化 → Stage 3: DPO/RLHF
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
API_BASE = "https://api.neuroq.he-ro.jp"

# 学習ステージの定義
TRAINING_STAGES = [
    {
        "stage": "Stage 1: 事前学習 (Pre-training)",
        "datasets": [
            {
                "id": "hotchpotch/fineweb-2-edu-japanese",
                "config": {"split": "train[:80%]"},
                "name": "FineWeb 2 EDU Japanese (80%)"
            },
            {
                "id": "range3/cc100-ja",
                "config": {"split": "train[:20%]"},
                "name": "CC100 Japanese (20%, 量重視補完)"
            }
        ],
        "params": {
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 5e-4,
            "warmup_steps": 1000,
            "model_size": "medium"
        },
        "description": "大規模日本語データセットで汎用知識の習得"
    },
    {
        "stage": "Stage 1B: 軽量代替 (オプション)",
        "datasets": [
            {
                "id": "ohtaman/oscar_ja_clean_filtered",
                "config": {},
                "name": "OSCAR Japanese Clean"
            },
            {
                "id": "izumi-lab/mc4-ja",
                "config": {"split": "train[:50%]"},
                "name": "MC4 Japanese (50%)"
            }
        ],
        "params": {
            "epochs": 2,
            "batch_size": 32,
            "learning_rate": 5e-4,
            "model_size": "small"
        },
        "description": "計算資源限定時の軽量代替オプション"
    },
    {
        "stage": "Stage 2: 会話化 (Conversation Fine-tuning)",
        "datasets": [
            {
                "id": "izumi-lab/llm-japanese-dataset",
                "config": {},
                "name": "LLM Japanese Dataset"
            },
            {
                "id": "llm-jp/magpie-sft-v1.0",
                "config": {},
                "name": "Magpie SFT v1.0"
            }
        ],
        "params": {
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 2e-4,
            "model_size": "medium"
        },
        "description": "指示従順性と会話能力の向上"
    },
    {
        "stage": "Stage 3: 仕上げ (DPO/RLHF)",
        "datasets": [
            {
                "id": "llm-jp/hh-rlhf-12k-ja",
                "config": {},
                "name": "HH-RLHF 12k Japanese"
            }
        ],
        "params": {
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "model_size": "medium"
        },
        "description": "高品質応答の強化学習（DPO/RLHF）"
    }
]


def submit_training_job(datasets: list, params: Dict[str, Any]) -> str:
    """
    訓練ジョブをサブミットし、call_idを返す
    """
    payload = {
        "dataset_ids": [d["id"] for d in datasets],
        "datasets_with_config": datasets,
        "epochs": params["epochs"],
        "batch_size": params.get("batch_size", 32),
        "learning_rate": params.get("learning_rate", 5e-4),
        "warmup_steps": params.get("warmup_steps", 500),
        "model_size": params.get("model_size", "medium"),
        "gradient_accumulation_steps": params.get("gradient_accumulation_steps", 2)
    }

    print(f"  📤 訓練リクエストを送信中...")
    print(f"     - Dataset数: {len(datasets)}")
    print(f"     - Epochs: {params['epochs']}")
    print(f"     - Model Size: {params['model_size']}")

    response = requests.post(f"{API_BASE}/train", json=payload)

    if response.status_code != 200:
        print(f"  ❌ エラー: {response.status_code}")
        print(f"     {response.text}")
        raise Exception(f"Training submission failed: {response.text}")

    result = response.json()
    call_id = result.get("call_id")
    print(f"  ✅ サブミット成功 - Call ID: {call_id}")
    return call_id


def check_training_status(call_id: str) -> Dict[str, Any]:
    """
    訓練ジョブのステータスを確認
    """
    response = requests.get(f"{API_BASE}/train/status/{call_id}")
    if response.status_code != 200:
        raise Exception(f"Failed to check status: {response.text}")
    return response.json()


def wait_for_training(call_id: str, max_wait_hours: int = 48) -> bool:
    """
    訓練完了を待機
    """
    start_time = time.time()
    max_wait_seconds = max_wait_hours * 3600
    check_interval = 60  # 1分ごとにチェック

    print(f"\n  ⏳ 訓練完了を待機中 (最大 {max_wait_hours} 時間)...")

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            print(f"  ⚠️ タイムアウト: {max_wait_hours} 時間経過")
            return False

        try:
            status = check_training_status(call_id)

            if status["status"] == "completed":
                print(f"  ✅ 訓練完了!")
                if "result" in status:
                    result = status["result"]
                    if isinstance(result, dict) and "best_val_loss" in result:
                        print(f"     Best Val Loss: {result['best_val_loss']:.4f}")
                return True

            elif status["status"] == "running":
                elapsed_min = int(elapsed / 60)
                print(f"     [進行中] {elapsed_min}分経過 - {status.get('message', 'Running')}")

            else:
                print(f"  ⚠️ ステータス: {status['status']}")
                if "error" in status:
                    print(f"     エラー: {status['error']}")
                    return False

        except Exception as e:
            print(f"     ⚠️ ステータス確認エラー: {e}")

        time.sleep(check_interval)


def show_learned_datasets():
    """
    学習済みデータセット一覧を表示
    """
    try:
        response = requests.get(f"{API_BASE}/learned_datasets")
        if response.status_code == 200:
            data = response.json()
            datasets = data.get("learned_datasets", [])

            if datasets:
                print("\n📚 学習済みデータセット一覧:")
                print("=" * 60)
                for i, ds in enumerate(datasets, 1):
                    print(f"\n{i}. モデルサイズ: {ds.get('model_size', 'N/A')}")
                    print(f"   学習時刻: {ds.get('trained_at', 'N/A')}")
                    print(f"   Best Val Loss: {ds.get('best_val_loss', 'N/A')}")
                    print(f"   Epochs: {ds.get('epochs', 'N/A')}")
                    if "datasets" in ds:
                        print(f"   データセット:")
                        for dataset in ds["datasets"]:
                            if isinstance(dataset, dict):
                                print(f"     - {dataset.get('id', dataset)}")
                            else:
                                print(f"     - {dataset}")
            else:
                print("\n📚 まだ学習済みデータセットがありません")
    except Exception as e:
        print(f"\n⚠️ 学習済みデータセット取得エラー: {e}")


def main():
    """
    メイン実行関数
    """
    print("\n" + "=" * 70)
    print("🚀 NeuroQ 推奨データセット学習パイプライン")
    print("=" * 70)

    # 実行オプションを確認
    print("\n実行モード:")
    print("1. Stage 1: 事前学習のみ")
    print("2. Stage 1B: 軽量代替のみ")
    print("3. Stage 2: 会話化のみ")
    print("4. Stage 3: DPO/RLHF仕上げのみ")
    print("5. Stage 1 → 2 → 3 (フル学習パイプライン)")
    print("6. 学習済みデータセット一覧確認")
    print("0. キャンセル")

    choice = input("\n選択 (0-6): ").strip()

    if choice == "0":
        print("キャンセルしました")
        return
    elif choice == "6":
        show_learned_datasets()
        return

    # ステージ選択
    stage_choices = {
        "1": [TRAINING_STAGES[0]],
        "2": [TRAINING_STAGES[1]],
        "3": [TRAINING_STAGES[2]],
        "4": [TRAINING_STAGES[3]],
        "5": TRAINING_STAGES[:3]  # Stage 1, 2, 3
    }

    stages = stage_choices.get(choice, [])
    if not stages:
        print("❌ 無効な選択です")
        return

    call_ids = []

    # 各ステージを実行
    for stage_info in stages:
        print(f"\n{'=' * 70}")
        print(f"📍 {stage_info['stage']}")
        print(f"{'=' * 70}")
        print(f"説明: {stage_info['description']}")
        print(f"\nデータセット:")
        for ds in stage_info["datasets"]:
            config_str = f" {json.dumps(ds['config'])}" if ds['config'] else ""
            print(f"  • {ds['name']}{config_str}")

        print(f"\nパラメータ:")
        for key, value in stage_info["params"].items():
            print(f"  • {key}: {value}")

        # 確認
        confirm = input("\n実行しますか? (y/n): ").strip().lower()
        if confirm != "y":
            print("  スキップしました")
            continue

        # ジョブサブミット
        call_id = submit_training_job(stage_info["datasets"], stage_info["params"])
        call_ids.append(call_id)

        # 待機
        if wait_for_training(call_id):
            print(f"\n✅ {stage_info['stage']} 完了!\n")
        else:
            print(f"\n❌ {stage_info['stage']} タイムアウト\n")
            # タイムアウト後も次のステージに進めるか確認
            cont = input("次のステージに進みますか? (y/n): ").strip().lower()
            if cont != "y":
                break

    # 最終結果
    print("\n" + "=" * 70)
    print("📊 学習パイプライン完了")
    print("=" * 70)
    print(f"\nサブミットされたジョブ数: {len(call_ids)}")
    if call_ids:
        print("Call IDs:")
        for i, cid in enumerate(call_ids, 1):
            print(f"  {i}. {cid}")

    # 最終状態を表示
    show_learned_datasets()

    print("\n✨ すべての学習が完了しました!")


if __name__ == "__main__":
    main()
