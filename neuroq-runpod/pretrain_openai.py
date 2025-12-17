#!/usr/bin/env python3
"""
NeuroQ OpenAIデータセット事前学習スクリプト
============================================
OpenAIの公開データセットを使用して学習を行います。

使用データセット:
- openai/mrcr (Multi-turn Reasoning Chain Retrieval)
- openai/openai_humaneval (HumanEval - コード生成評価)
- openai/MMMLU (Multilingual MMLU - 多言語知識評価)

使い方:
    pip install datasets
    python pretrain_openai.py

生成されるファイル:
    - neuroq_pretrained.pt (学習済みモデル)
"""

import torch
import os
import sys
from typing import List, Optional

print("=" * 60)
print("🧠 NeuroQ OpenAIデータセット事前学習")
print("=" * 60)

# Hugging Face datasets をインポート
try:
    from datasets import load_dataset
    print("✅ datasets ライブラリをインポートしました")
except ImportError:
    print("❌ datasets ライブラリがインストールされていません")
    print("   実行: pip install datasets")
    sys.exit(1)

# neuroquantum_layered.py をインポート
from neuroquantum_layered import NeuroQuantumAI


def load_mrcr_data() -> List[str]:
    """
    openai/mrcr データセットをロード
    Multi-turn Reasoning Chain Retrieval - 推論チェーンデータ
    """
    print("\n📥 openai/mrcr データセットをロード中...")
    texts = []
    
    try:
        dataset = load_dataset("openai/mrcr", split="test")
        print(f"   ✅ {len(dataset)} サンプルをロードしました")
        
        for idx, item in enumerate(dataset):
            if idx >= 100:  # 最大100サンプルに制限（日本語データ優先のため）
                break
            # 様々なフィールドからテキストを抽出
            if "haystack" in item and item["haystack"]:
                for doc in item["haystack"][:3]:  # 最初の3つに制限
                    if isinstance(doc, str) and len(doc) > 20:
                        texts.append(doc[:300])
            
            if "needle" in item and item["needle"]:
                needle = item["needle"]
                if isinstance(needle, str) and len(needle) > 10:
                    texts.append(needle)
            
            if "question" in item and item["question"]:
                q = item["question"]
                if isinstance(q, str):
                    texts.append(f"質問: {q}")
            
            if "answer" in item and item["answer"]:
                a = item["answer"]
                if isinstance(a, str):
                    texts.append(f"回答: {a}")
        
        print(f"   ✅ mrcr から {len(texts)} テキストを抽出")
        
    except Exception as e:
        print(f"   ⚠️ mrcr ロードエラー: {e}")
    
    return texts


def load_humaneval_data() -> List[str]:
    """
    openai/openai_humaneval データセットをロード
    HumanEval - コード生成評価データセット
    """
    print("\n📥 openai/openai_humaneval データセットをロード中...")
    texts = []
    
    try:
        dataset = load_dataset("openai/openai_humaneval", split="test")
        print(f"   ✅ {len(dataset)} サンプルをロードしました")
        
        for item in dataset:
            # プロンプト（関数の説明）
            if "prompt" in item and item["prompt"]:
                prompt = item["prompt"]
                texts.append(f"プログラミング課題:\n{prompt}")
            
            # 正規解答
            if "canonical_solution" in item and item["canonical_solution"]:
                solution = item["canonical_solution"]
                texts.append(f"解答コード:\n{solution}")
            
            # タスクID と説明を組み合わせ
            if "task_id" in item and "prompt" in item:
                task_text = f"タスク {item['task_id']}: {item['prompt'][:200]}"
                texts.append(task_text)
        
        print(f"   ✅ humaneval から {len(texts)} テキストを抽出")
        
    except Exception as e:
        print(f"   ⚠️ humaneval ロードエラー: {e}")
    
    return texts


def load_mmmlu_data() -> List[str]:
    """
    openai/MMMLU データセットをロード
    Multilingual MMLU - 多言語知識評価データセット
    """
    print("\n📥 openai/MMMLU データセットをロード中...")
    texts = []
    
    try:
        # 日本語サブセットをロード
        dataset = load_dataset("openai/MMMLU", "JA_JP", split="test")
        print(f"   ✅ MMMLU日本語: {len(dataset)} サンプルをロードしました")
        
        for item in dataset:
            # 質問
            if "Question" in item and item["Question"]:
                q = item["Question"]
                
                # 選択肢を取得
                choices = []
                for key in ["A", "B", "C", "D"]:
                    if key in item and item[key]:
                        choices.append(f"{key}. {item[key]}")
                
                # 正解
                answer = item.get("Answer", "")
                
                # Q&A形式でテキスト作成
                if choices:
                    choices_text = "\n".join(choices)
                    qa_text = f"問題: {q}\n選択肢:\n{choices_text}\n正解: {answer}"
                    texts.append(qa_text)
                
                # 質問と回答のペア
                if answer and answer in ["A", "B", "C", "D"]:
                    answer_text = item.get(answer, "")
                    if answer_text:
                        texts.append(f"<USER>{q}<ASSISTANT>{answer_text}")
            
            # Subject（科目）も追加
            if "Subject" in item and item["Subject"]:
                subject = item["Subject"]
                if "Question" in item:
                    texts.append(f"科目「{subject}」の問題: {item['Question']}")
        
        print(f"   ✅ MMMLU から {len(texts)} テキストを抽出")
        
    except Exception as e:
        print(f"   ⚠️ MMMLU ロードエラー: {e}")
        
        # フォールバック: 英語版を試す
        try:
            print("   📥 英語版MMMMLUを試行中...")
            dataset = load_dataset("openai/MMMLU", "EN_US", split="test")
            print(f"   ✅ MMMLU英語: {len(dataset)} サンプルをロードしました")
            
            for item in dataset:
                if "Question" in item and item["Question"]:
                    q = item["Question"]
                    answer = item.get("Answer", "")
                    if answer and answer in ["A", "B", "C", "D"]:
                        answer_text = item.get(answer, "")
                        if answer_text:
                            # 英語Q&Aも追加
                            texts.append(f"Question: {q}\nAnswer: {answer_text}")
            
            print(f"   ✅ MMMLU英語から {len(texts)} テキストを抽出")
            
        except Exception as e2:
            print(f"   ⚠️ MMMLU英語ロードエラー: {e2}")
    
    return texts


def generate_japanese_instruction_data() -> List[str]:
    """日本語指示データを生成"""
    
    instruction_data = [
        # ChatGPTについて
        "<USER>ChatGPTについて教えて<ASSISTANT>ChatGPTはOpenAIが開発した大規模言語モデルです。GPTはGenerative Pre-trained Transformerの略で、数十億のパラメータを持つニューラルネットワークです。自然言語で人間と会話でき、質問応答、文章作成、翻訳、コード生成など様々なタスクをこなせます。",
        "<USER>ChatGPTとは何ですか？<ASSISTANT>ChatGPTはOpenAIが作った会話型AIです。大量のテキストデータで学習し、人間のような自然な文章を生成できます。2022年11月に公開され、世界中で大きな注目を集めています。",
        "<USER>GPTの仕組みを教えて<ASSISTANT>GPTはTransformerアーキテクチャをベースにした言語モデルです。大量のテキストデータで事前学習し、次の単語を予測する能力を獲得します。アテンション機構により、文脈を理解して適切な応答を生成します。",
        
        # 量子コンピュータについて
        "<USER>量子コンピュータについて教えて<ASSISTANT>量子コンピュータは量子力学の原理を利用した次世代の計算機です。量子ビットは0と1の重ね合わせ状態を取ることができ、従来のコンピュータでは解けない問題を高速に解ける可能性があります。暗号解読、創薬、最適化問題などで革新的な応用が期待されています。",
        "<USER>量子コンピュータとは？<ASSISTANT>量子コンピュータは量子もつれや重ね合わせといった量子力学の性質を計算に利用するコンピュータです。GoogleやIBMなどが開発を進めており、将来的には暗号解読や新薬開発で威力を発揮すると期待されています。",
        "<USER>量子ビットとは？<ASSISTANT>量子ビットは量子コンピュータの基本単位です。古典的なビットが0か1のどちらかの状態しか取れないのに対し、量子ビットは0と1の重ね合わせ状態を取ることができます。これにより、並列計算が可能になります。",
        
        # AIについて
        "<USER>人工知能について教えて<ASSISTANT>人工知能（AI）は人間の知能を模倣するコンピュータシステムです。機械学習やディープラーニングといった技術を使い、画像認識、音声認識、自然言語処理など様々な分野で活用されています。近年はChatGPTのような大規模言語モデルが注目を集めています。",
        "<USER>AIとは何ですか？<ASSISTANT>AI（人工知能）は人間の知能的な振る舞いをコンピュータで再現する技術です。学習、推論、問題解決などの能力を持ち、自動運転、医療診断、翻訳など幅広い分野で応用されています。",
        "<USER>機械学習とは？<ASSISTANT>機械学習はデータからパターンを学習するAIの手法です。大量のデータを与えることで、コンピュータが自動的に規則性を見つけ出します。教師あり学習、教師なし学習、強化学習などの種類があります。",
        "<USER>ディープラーニングとは？<ASSISTANT>ディープラーニングは多層のニューラルネットワークを使った機械学習の手法です。画像認識や自然言語処理で高い性能を発揮し、現代のAI技術の中核を担っています。",
        
        # ニューロQについて
        "<USER>ニューロQとは何ですか？<ASSISTANT>ニューロQは量子ビットニューラルネットワーク（QBNN）を使った次世代の人工知能です。量子コンピュータの原理を活用し、従来のニューラルネットワークとは異なるアプローチで学習と推論を行います。量子もつれを活用した高効率な計算が特徴です。",
        "<USER>ニューロQについて教えて<ASSISTANT>ニューロQは日本で開発された量子生成AIです。QBNNという独自のアーキテクチャを使用し、Transformerモデルに量子もつれの概念を導入しています。従来のモデルとは異なる特性を持ち、新しいAIの可能性を探求しています。",
        "<USER>QBNNとは？<ASSISTANT>QBNN（Quantum Bit Neural Network）は量子ビットの概念をニューラルネットワークに応用したアーキテクチャです。量子もつれやエンタングルメントの特性を模倣し、効率的な学習と推論を実現します。",
        
        # プログラミングについて
        "<USER>Pythonについて教えて<ASSISTANT>PythonはAI開発で最も人気のあるプログラミング言語です。シンプルな文法で読みやすく、豊富なライブラリがあります。機械学習にはPyTorchやTensorFlow、データ分析にはpandasやnumpyがよく使われます。",
        "<USER>プログラミングとは？<ASSISTANT>プログラミングはコンピュータに指示を与えるためのコードを書く作業です。Python、JavaScript、Java、C++など様々な言語があり、それぞれ得意な分野があります。論理的思考力と問題解決能力が重要です。",
        "<USER>機械学習のライブラリは？<ASSISTANT>機械学習で人気のライブラリには、PyTorch、TensorFlow、scikit-learn、Kerasなどがあります。PyTorchは研究用途で人気があり、TensorFlowは本番環境での利用が多いです。",
        
        # 挨拶（多様なパターン）
        "<USER>こんにちは<ASSISTANT>こんにちは！私はニューロQです。何かお手伝いできることはありますか？",
        "<USER>こんにちは<ASSISTANT>こんにちは！ご質問があればお気軽にどうぞ。",
        "<USER>こんにちは<ASSISTANT>こんにちは！今日は何をお手伝いしましょうか？",
        "<USER>こんにちは！<ASSISTANT>こんにちは！元気ですか？何か知りたいことがあれば聞いてください。",
        "<USER>おはよう<ASSISTANT>おはようございます！今日も一日頑張りましょう。",
        "<USER>おはようございます<ASSISTANT>おはようございます！良い一日になりますように。何かお手伝いできることはありますか？",
        "<USER>こんばんは<ASSISTANT>こんばんは！夜遅くまでお疲れ様です。",
        "<USER>こんばんは<ASSISTANT>こんばんは！何かご質問はありますか？",
        "<USER>ありがとう<ASSISTANT>どういたしまして！お役に立てて嬉しいです。",
        "<USER>ありがとうございます<ASSISTANT>こちらこそありがとうございます！他に何かあればお聞きください。",
        "<USER>さようなら<ASSISTANT>さようなら！またお話できることを楽しみにしています。",
        "<USER>バイバイ<ASSISTANT>バイバイ！良い一日を！",
        "<USER>お疲れ様<ASSISTANT>お疲れ様です！ゆっくり休んでくださいね。",
        "<USER>助けて<ASSISTANT>はい、どうしましたか？お困りのことがあればお聞かせください。",
        "<USER>教えて<ASSISTANT>はい、何について知りたいですか？",
        "<USER>質問があります<ASSISTANT>はい、どうぞ！何でも聞いてください。",
        "<USER>ねえ<ASSISTANT>はい、何ですか？",
        "<USER>すみません<ASSISTANT>はい、どうしましたか？",
        "<USER>お願いします<ASSISTANT>はい、何をお手伝いしましょうか？",
        "<USER>調子はどう？<ASSISTANT>元気ですよ！ありがとうございます。あなたはいかがですか？",
        "<USER>元気？<ASSISTANT>はい、元気です！あなたも元気そうで何よりです。",
        
        # 一般的な質問
        "<USER>日本の首都はどこですか？<ASSISTANT>日本の首都は東京です。東京は政治、経済、文化の中心地で、人口は約1,400万人です。1868年の明治維新で京都から遷都されました。",
        "<USER>富士山の高さは？<ASSISTANT>富士山の標高は3,776メートルです。日本で最も高い山であり、世界文化遺産にも登録されています。美しい円錐形の姿は日本の象徴として知られています。",
        "<USER>地球は何歳ですか？<ASSISTANT>地球は約46億年前に形成されたと考えられています。太陽系が形成される過程で、ガスや塵が集まって地球が誕生しました。",
        "<USER>光の速さは？<ASSISTANT>光の速さは秒速約30万キロメートル（正確には299,792,458 m/s）です。これは物理学における最高速度であり、相対性理論の基礎となっています。",
        "<USER>水の化学式は？<ASSISTANT>水の化学式はH2Oです。水素原子2つと酸素原子1つから構成されています。水は生命にとって不可欠な物質です。",
        
        # 技術的な質問
        "<USER>APIとは何ですか？<ASSISTANT>API（Application Programming Interface）はソフトウェア同士が通信するためのインターフェースです。例えば、天気予報アプリが気象データを取得したり、支払いシステムと連携したりする際に使用されます。",
        "<USER>クラウドコンピューティングとは？<ASSISTANT>クラウドコンピューティングはインターネット経由でコンピューティングリソースを利用するサービスです。AWS、Google Cloud、Azureなどが代表的なプロバイダーで、サーバー、ストレージ、データベースなどを柔軟に利用できます。",
        "<USER>ブロックチェーンとは？<ASSISTANT>ブロックチェーンは分散型台帳技術です。取引履歴を暗号化してブロックに記録し、チェーン状に連結します。ビットコインなどの暗号通貨の基盤技術として知られ、改ざんが困難な特性があります。",
    ]
    
    return instruction_data


def generate_knowledge_data() -> List[str]:
    """一般知識データを生成"""
    
    knowledge = [
        # 科学
        "科学は観察と実験に基づく知識体系です。仮説を立て、実験で検証し、理論を構築します。",
        "物理学は自然界の法則を研究する学問です。力学、電磁気学、量子力学などの分野があります。",
        "化学は物質の性質と変化を研究する学問です。原子や分子の構造、化学反応を扱います。",
        "生物学は生命の仕組みを研究する学問です。細胞、遺伝子、進化などを扱います。",
        "数学は論理と抽象的な構造を研究する学問です。代数、幾何学、解析学などの分野があります。",
        
        # 技術
        "コンピュータサイエンスは計算と情報を研究する学問です。アルゴリズム、データ構造、プログラミングを扱います。",
        "インターネットは世界中のコンピュータを接続するネットワークです。ウェブサイト、メール、動画配信などに使用されます。",
        "スマートフォンは携帯電話とコンピュータの機能を組み合わせたデバイスです。アプリを通じて様々なサービスを利用できます。",
        "5Gは第5世代移動通信システムです。高速・大容量・低遅延の通信を実現し、IoTや自動運転に活用されます。",
        
        # AI・機械学習
        "ニューラルネットワークは脳の神経回路を模倣した計算モデルです。入力層、隠れ層、出力層から構成されます。",
        "トランスフォーマーは2017年に発表されたニューラルネットワークアーキテクチャです。アテンション機構を使い、自然言語処理で高い性能を発揮します。",
        "GPTはGenerative Pre-trained Transformerの略です。大量のテキストで事前学習し、様々なタスクに適用できます。",
        "BERTはBidirectional Encoder Representations from Transformersの略です。双方向の文脈を理解する言語モデルです。",
        "強化学習は試行錯誤から学ぶ機械学習の手法です。報酬を最大化するように行動を学習します。",
        
        # 量子コンピュータ
        "量子もつれは二つの量子が相関を持つ現象です。一方の状態を測定すると、他方の状態も決まります。",
        "量子重ね合わせは量子が複数の状態を同時に取る現象です。これにより並列計算が可能になります。",
        "量子ゲートは量子ビットを操作する基本単位です。パウリゲート、アダマールゲートなどがあります。",
        "量子誤り訂正は量子計算のエラーを修正する技術です。量子コンピュータの実用化に不可欠です。",
        "超電導量子ビットはジョセフソン接合を使った量子ビットです。IBMやGoogleが採用しています。",
    ]
    
    return knowledge * 20  # 20倍に増やす


def main():
    # デバイス確認
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("💻 CPU")
    
    # データセットをロード
    all_texts = []
    
    # 1. openai/mrcr - 推論チェーンデータ
    mrcr_texts = load_mrcr_data()
    all_texts.extend(mrcr_texts)
    print(f"✅ openai/mrcr: {len(mrcr_texts)} テキスト追加")
    
    # 2. openai/openai_humaneval
    humaneval_texts = load_humaneval_data()
    all_texts.extend(humaneval_texts)
    
    # 3. openai/MMMLU (Multilingual MMLU)
    mmmlu_texts = load_mmmlu_data()
    all_texts.extend(mmmlu_texts)
    
    # 4. 日本語指示データ（最優先・大量に追加）
    print("\n📚 日本語指示データを追加中...")
    instruction_texts = generate_japanese_instruction_data()
    all_texts.extend(instruction_texts * 500)  # 500倍に増やす（最重要）
    
    # 5. 一般知識データ（増量）
    print("📚 一般知識データを追加中...")
    knowledge_texts = generate_knowledge_data()
    all_texts.extend(knowledge_texts * 10)  # 10倍に増やす
    
    print(f"\n📊 最終学習データ:")
    print(f"   総テキスト数: {len(all_texts):,}")
    print(f"   総文字数: {sum(len(s) for s in all_texts):,}")
    
    # モデル作成
    print("\n🧠 モデル作成中...")
    model = NeuroQuantumAI(
        embed_dim=256,
        hidden_dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=256,
        dropout=0.1,
        lambda_entangle=0.5
    )
    
    # 学習
    print("\n🚀 学習開始...")
    print("   （これには10分〜30分かかる場合があります）")
    
    model.train(
        all_texts,
        epochs=25,
        seq_len=128,
        batch_size=16,
        lr=0.0005
    )
    
    # モデル保存
    save_path = "neuroq_pretrained.pt"
    print(f"\n💾 モデル保存中: {save_path}")
    
    save_data = {
        'model_state_dict': model.model.state_dict(),
        'config': {
            'vocab_size': model.config.vocab_size,
            'embed_dim': model.config.embed_dim,
            'hidden_dim': model.config.hidden_dim,
            'num_heads': model.config.num_heads,
            'num_layers': model.config.num_layers,
            'max_seq_len': model.config.max_seq_len,
            'dropout': model.config.dropout,
            'lambda_entangle': model.config.lambda_entangle,
        },
        'tokenizer_vocab_size': model.tokenizer.actual_vocab_size or model.tokenizer.vocab_size,
        'training_info': {
            'datasets': ['openai/mrcr', 'openai/openai_humaneval', 'openai/MMMLU', 'japanese_instructions', 'knowledge'],
            'total_texts': len(all_texts),
            'epochs': 25,
        }
    }
    
    torch.save(save_data, save_path)
    
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✅ 保存完了: {save_path} ({file_size:.1f} MB)")
    
    # テスト生成
    print("\n🧪 生成テスト:")
    test_prompts = [
        "ChatGPTについて教えて",
        "量子コンピュータとは",
        "ニューロQとは",
        "こんにちは",
        "Pythonについて教えて",
    ]
    
    for prompt in test_prompts:
        result = model.generate(prompt, max_length=100, temperature=0.7)
        print(f"\n   入力: {prompt}")
        print(f"   出力: {result[:200]}...")
    
    print("\n" + "=" * 60)
    print("✅ OpenAIデータセット事前学習完了！")
    print("=" * 60)
    print("\n次のステップ:")
    print("1. git add neuroq_pretrained.pt")
    print("2. git commit -m 'Update pretrained model with OpenAI datasets'")
    print("3. git push origin main")
    print("4. RunPodでRebuild")


if __name__ == "__main__":
    main()
