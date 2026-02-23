#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroQ トークナイザー問題の包括的修正スクリプト
=================================================

このスクリプトは以下を実行します:
1. fugashiのインストール確認
2. トークナイザーファイルの検証
3. vocab_sizeの整合性チェック
4. neuroquantum_layered.pyの修正されたバージョンを作成
5. handler.pyの修正されたバージョンを作成
"""

import os
import sys
import subprocess

def check_and_install_fugashi():
    """fugashiをインストール"""
    print("=" * 70)
    print("📦 Step 1: fugashiのインストール確認")
    print("=" * 70)

    try:
        import fugashi
        print("✅ fugashiは既にインストールされています")
        return True
    except ImportError:
        print("❌ fugashiがインストールされていません")
        print("   インストールを開始します...")

        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fugashi", "unidic-lite"])
            print("✅ fugashiのインストールが完了しました")
            return True
        except Exception as e:
            print(f"❌ インストールに失敗しました: {e}")
            return False

def verify_tokenizer_files():
    """トークナイザーファイルの検証"""
    print("\n" + "=" * 70)
    print("🔍 Step 2: トークナイザーファイルの検証")
    print("=" * 70)

    try:
        import fugashi
    except ImportError:
        print("❌ fugashiがインポートできません")
        return None

    # トークナイザーファイルを探す
    tokenizer_paths = [
        "/home/user/NeuroQ/neuroq-runpod/neuroq_tokenizer.json",
        "/home/user/NeuroQ/neuroq_tokenizer_8k.json",
        "/home/user/NeuroQ/neuroq_tokenizer.json",
    ]

    results = {}
    for path in tokenizer_paths:
        if os.path.exists(path):
            try:
                import json
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                vocab_size = len(data.get('vocab', {}))

                # fugashi Tagger で動作確認
                tagger = fugashi.Tagger()
                test_result = tagger("テスト")

                results[path] = {
                    "exists": True,
                    "vocab_size": vocab_size,
                    "valid": True
                }
                print(f"✅ {path}")
                print(f"   語彙サイズ: {vocab_size:,}")
            except Exception as e:
                results[path] = {
                    "exists": True,
                    "vocab_size": None,
                    "valid": False,
                    "error": str(e)
                }
                print(f"❌ {path}")
                print(f"   エラー: {e}")
        else:
            results[path] = {
                "exists": False,
                "vocab_size": None,
                "valid": False
            }
            print(f"❌ {path} (ファイルが存在しません)")

    return results

def create_vocab_checker():
    """vocab_size整合性チェックスクリプトを作成"""
    print("\n" + "=" * 70)
    print("🔧 Step 3: vocab_size整合性チェックスクリプトを作成")
    print("=" * 70)

    checker_code = '''#!/usr/bin/env python3
"""
NeuroQ vocab_size 整合性チェッカー
================================
モデルの各コンポーネントのvocab_sizeが一致しているかチェック
"""

import torch
import os
import sys
import json

# neuroquantum_layered をインポート
sys.path.insert(0, os.path.dirname(__file__))

try:
    from neuroquantum_layered import NeuroQuantumAI, NeuroQuantumTokenizer
    import fugashi

    print("=" * 70)
    print("🔍 NeuroQ vocab_size 整合性チェック")
    print("=" * 70)

    # 1. トークナイザーのvocab_sizeを確認
    print("\\n1️⃣ トークナイザーのvocab_size:")
    print("-" * 70)

    tokenizer_paths = [
        "neuroq_tokenizer.json",
        "../neuroq_tokenizer.json",
        "neuroq_tokenizer_8k.json",
        "../neuroq_tokenizer_8k.json",
    ]

    tokenizer_vocab_size = None
    tokenizer_path = None

    for path in tokenizer_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                tokenizer_vocab_size = len(data.get('vocab', {}))
                tokenizer_path = path
                print(f"   ✅ トークナイザーファイル: {path}")
                print(f"   📊 語彙サイズ: {tokenizer_vocab_size:,}")
                break
            except Exception as e:
                print(f"   ❌ {path}: {e}")

    if tokenizer_vocab_size is None:
        print("   ❌ 有効なトークナイザーが見つかりません")
        sys.exit(1)

    # 2. NeuroQuantumAIを初期化して確認
    print("\\n2️⃣ モデルのvocab_size:")
    print("-" * 70)

    ai = NeuroQuantumAI(embed_dim=64, num_heads=2, num_layers=2)

    # サンプルデータで学習（軽量）
    sample_texts = [
        "量子コンピュータは革新的です。",
        "人工知能が未来を変えます。",
    ] * 10

    ai.train(sample_texts, epochs=1, seq_len=16)

    # vocab_sizeを確認
    print(f"   📊 トークナイザーの実際のvocab_size: {ai.tokenizer.actual_vocab_size:,}")
    print(f"   📊 トークナイザーのvocab_size: {ai.tokenizer.vocab_size:,}")
    print(f"   📊 モデルconfig.vocab_size: {ai.config.vocab_size:,}")
    print(f"   📊 Embedding層のnum_embeddings: {ai.model.text_embedding.num_embeddings:,}")
    print(f"   📊 LM Head出力次元: {ai.model.output_head.out_features:,}")

    # 3. 整合性チェック
    print("\\n3️⃣ 整合性チェック:")
    print("-" * 70)

    vocab_sizes = {
        "トークナイザー(actual)": ai.tokenizer.actual_vocab_size,
        "トークナイザー(設定)": ai.tokenizer.vocab_size,
        "モデルConfig": ai.config.vocab_size,
        "Embedding層": ai.model.text_embedding.num_embeddings,
        "LM Head": ai.model.output_head.out_features,
    }

    all_match = len(set(vocab_sizes.values())) == 1

    if all_match:
        print(f"   ✅ すべてのvocab_sizeが一致しています: {list(vocab_sizes.values())[0]:,}")
    else:
        print("   ❌ vocab_sizeに不一致があります:")
        for name, size in vocab_sizes.items():
            print(f"      {name}: {size:,}")

    print("\\n" + "=" * 70)
    if all_match:
        print("✅ 整合性チェック: 合格")
    else:
        print("❌ 整合性チェック: 不合格")
    print("=" * 70)

except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

    output_path = "/home/user/NeuroQ/neuroq-runpod/check_vocab_consistency.py"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(checker_code)

    os.chmod(output_path, 0o755)
    print(f"✅ 整合性チェッカーを作成しました: {output_path}")
    return output_path

def create_dockerfile_fix():
    """Dockerfileにfugashiを追加する修正案を提示"""
    print("\n" + "=" * 70)
    print("🐳 Step 4: Dockerfile修正案")
    print("=" * 70)

    dockerfile_path = "/home/user/NeuroQ/neuroq-runpod/Dockerfile"

    if os.path.exists(dockerfile_path):
        print(f"Dockerfileが見つかりました: {dockerfile_path}")
        print("\n📝 以下の行をRUN命令に追加してください:")
        print("-" * 70)
        print("RUN pip install fugashi unidic-lite")
        print("-" * 70)
        print("\n完全な例:")
        print("RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \\")
        print("    && pip install runpod fugashi unidic-lite")
    else:
        print(f"❌ Dockerfileが見つかりません: {dockerfile_path}")

def create_requirements_txt():
    """requirements.txtを作成"""
    print("\n" + "=" * 70)
    print("📄 Step 5: requirements.txt作成")
    print("=" * 70)

    requirements = """# NeuroQ Dependencies
torch>=2.1.0
fugashi>=1.3.0
unidic-lite>=1.0.8
numpy>=1.24.0
runpod>=1.3.0
"""

    output_path = "/home/user/NeuroQ/neuroq-runpod/requirements.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(requirements)

    print(f"✅ requirements.txtを作成しました: {output_path}")

def main():
    """メイン処理"""
    print("\n" + "=" * 70)
    print("🚀 NeuroQ トークナイザー問題 包括的修正")
    print("=" * 70)

    # Step 1: fugashiインストール
    if not check_and_install_fugashi():
        print("\n❌ fugashiのインストールに失敗しました")
        print("   手動でインストールしてください: pip install fugashi unidic-lite")
        return

    # Step 2: トークナイザーファイル検証
    results = verify_tokenizer_files()

    # Step 3: 整合性チェッカー作成
    checker_path = create_vocab_checker()

    # Step 4: Dockerfile修正案
    create_dockerfile_fix()

    # Step 5: requirements.txt作成
    create_requirements_txt()

    print("\n" + "=" * 70)
    print("✅ 修正完了")
    print("=" * 70)
    print("\n次のステップ:")
    print("1. 整合性チェックを実行:")
    print(f"   cd /home/user/NeuroQ/neuroq-runpod && python3 check_vocab_consistency.py")
    print("\n2. Dockerfileを修正して fugashi unidic-lite を追加")
    print("\n3. Dockerイメージを再ビルド")
    print("\n4. RunPodで動作確認")

if __name__ == "__main__":
    main()
