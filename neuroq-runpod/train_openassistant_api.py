#!/usr/bin/env python3
"""
OpenAssistant Conversations API学習スクリプト
==============================================

Hugging Face Datasets APIを使用してOpenAssistant Conversationsデータセットを
ダウンロードし、NeuroQモデルを学習します。

データセット: OpenAssistant/oasst1, OpenAssistant/oasst2

使い方:
    # 基本的な使い方（デフォルト設定）
    python train_openassistant_api.py

    # モデルサイズ指定
    python train_openassistant_api.py --model-size small
    python train_openassistant_api.py --model-size large

    # カスタム設定
    python train_openassistant_api.py --epochs 20 --batch-size 16 --max-samples 5000

    # 言語フィルタ（英語のみ）
    python train_openassistant_api.py --language en

    # 日本語と英語
    python train_openassistant_api.py --language ja,en

    # OASST2データセットを使用
    python train_openassistant_api.py --dataset oasst2
"""

import os
import sys
import torch
import logging
import random
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 現在のディレクトリをパスに追加
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


@dataclass
class TrainingConfig:
    """学習設定"""
    model_size: str = 'micro'
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 0.0005
    seq_length: int = 128
    max_samples: int = 5000
    languages: List[str] = None
    dataset_name: str = 'oasst1'
    checkpoint_dir: str = 'checkpoints'

    def __post_init__(self):
        if self.languages is None:
            self.languages = ['en']  # デフォルトは英語

        # モデルサイズに応じた設定調整
        if self.model_size == 'small':
            self.batch_size = min(self.batch_size, 4)
            self.learning_rate = 0.0003
        elif self.model_size == 'large':
            self.batch_size = min(self.batch_size, 2)
            self.learning_rate = 0.0002


class OpenAssistantDataLoader:
    """OpenAssistant Conversationsデータセットローダー"""

    SUPPORTED_DATASETS = {
        'oasst1': 'OpenAssistant/oasst1',
        'oasst2': 'OpenAssistant/oasst2',
    }

    def __init__(self, dataset_name: str = 'oasst1'):
        self.dataset_name = dataset_name
        self.dataset_id = self.SUPPORTED_DATASETS.get(dataset_name, dataset_name)
        self.messages_by_id: Dict = {}

    def load_dataset(self, split: str = 'train') -> 'Dataset':
        """Hugging Face APIからデータセットをロード"""
        from datasets import load_dataset

        logger.info(f"Hugging Face APIからデータセットをロード: {self.dataset_id}")

        try:
            dataset = load_dataset(self.dataset_id, split=split)
            logger.info(f"ロード完了: {len(dataset):,} メッセージ")
            return dataset
        except Exception as e:
            logger.error(f"データセットのロードに失敗: {e}")
            raise

    def build_conversation_tree(self, dataset) -> None:
        """メッセージをIDでインデックス化して会話ツリーを構築"""
        logger.info("会話ツリーを構築中...")

        self.messages_by_id = {}
        for msg in dataset:
            msg_id = msg.get('message_id', '')
            if msg_id:
                self.messages_by_id[msg_id] = {
                    'text': msg.get('text', ''),
                    'role': msg.get('role', ''),
                    'parent_id': msg.get('parent_id'),
                    'lang': msg.get('lang', ''),
                    'rank': msg.get('rank'),
                }

        logger.info(f"メッセージ数: {len(self.messages_by_id):,}")

    def extract_conversations(
        self,
        languages: List[str] = None,
        max_samples: int = None,
        include_multiturn: bool = True
    ) -> List[str]:
        """
        会話ペアを抽出

        Args:
            languages: フィルタする言語リスト（例: ['en', 'ja']）
            max_samples: 最大サンプル数
            include_multiturn: マルチターン会話を含めるか

        Returns:
            <USER>...<ASSISTANT>形式の会話リスト
        """
        logger.info(f"会話を抽出中... (言語: {languages or 'all'})")

        conversations = []
        lang_stats = {}

        # ルートメッセージを見つける
        root_messages = [
            (msg_id, msg) for msg_id, msg in self.messages_by_id.items()
            if msg['parent_id'] is None and msg['role'] == 'prompter'
        ]

        logger.info(f"ルートメッセージ数: {len(root_messages):,}")

        for root_id, root_msg in root_messages:
            if max_samples and len(conversations) >= max_samples:
                break

            # 言語フィルタ
            if languages and root_msg['lang'] not in languages:
                continue

            # 子メッセージ（assistantの返答）を探す
            children = self._find_children(root_id, 'assistant')

            if children:
                # ランクが高い返答を選択
                children.sort(key=lambda x: x[1].get('rank') or 999)
                response_id, response_msg = children[0]

                user_text = root_msg['text'].strip()
                assistant_text = response_msg['text'].strip()

                # 長さチェック
                if len(user_text) > 5 and len(assistant_text) > 10:
                    # 長すぎる場合はトリミング
                    user_text = user_text[:500]
                    assistant_text = assistant_text[:1000]

                    conversation = f"<USER> {user_text} <ASSISTANT> {assistant_text}"
                    conversations.append(conversation)

                    # 言語統計
                    lang = root_msg['lang']
                    lang_stats[lang] = lang_stats.get(lang, 0) + 1

                    # マルチターン会話の抽出
                    if include_multiturn and max_samples and len(conversations) < max_samples:
                        multiturn = self._extract_multiturn(response_id, languages)
                        conversations.extend(multiturn[:max_samples - len(conversations)])

        logger.info(f"抽出完了: {len(conversations):,} 会話")
        logger.info(f"言語分布: {lang_stats}")

        return conversations

    def _find_children(self, parent_id: str, role: str = None) -> List[Tuple[str, Dict]]:
        """指定された親IDの子メッセージを探す"""
        children = []
        for msg_id, msg in self.messages_by_id.items():
            if msg['parent_id'] == parent_id:
                if role is None or msg['role'] == role:
                    children.append((msg_id, msg))
        return children

    def _extract_multiturn(self, start_id: str, languages: List[str] = None) -> List[str]:
        """マルチターン会話を抽出"""
        multiturn_conversations = []
        current_id = start_id
        conversation_parts = []

        while current_id and len(multiturn_conversations) < 10:  # 最大10ターン
            # 次のユーザーメッセージを探す
            user_children = self._find_children(current_id, 'prompter')
            if not user_children:
                break

            # 言語フィルタ
            if languages:
                user_children = [(id, msg) for id, msg in user_children if msg['lang'] in languages]

            if not user_children:
                break

            user_id, user_msg = user_children[0]

            # そのユーザーメッセージへの返答を探す
            assistant_children = self._find_children(user_id, 'assistant')
            if not assistant_children:
                break

            assistant_children.sort(key=lambda x: x[1].get('rank') or 999)
            assistant_id, assistant_msg = assistant_children[0]

            user_text = user_msg['text'].strip()[:500]
            assistant_text = assistant_msg['text'].strip()[:1000]

            if len(user_text) > 5 and len(assistant_text) > 10:
                conversation = f"<USER> {user_text} <ASSISTANT> {assistant_text}"
                multiturn_conversations.append(conversation)

            current_id = assistant_id

        return multiturn_conversations


class NeuroQTrainer:
    """NeuroQモデルトレーナー"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        self.model_config = None

    def _get_device(self) -> torch.device:
        """最適なデバイスを選択"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"NVIDIA GPU使用: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Apple Silicon GPU (MPS) 使用")
        else:
            device = torch.device("cpu")
            logger.info("CPU使用")
        return device

    def initialize_model(self):
        """モデルとトークナイザーを初期化"""
        from tiktoken_tokenizer import TikTokenTokenizer

        logger.info(f"モデル初期化: {self.config.model_size.upper()}")

        # トークナイザー
        self.tokenizer = TikTokenTokenizer(encoding_name="o200k_base")
        logger.info(f"トークナイザー語彙サイズ: {self.tokenizer.vocab_size:,}")

        # モデル設定
        try:
            from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
            from model_configs import get_model_config, AVAILABLE_MODELS

            if self.config.model_size in ['small', 'large']:
                self.model_config = get_model_config(self.config.model_size)
            else:
                # Microモデル
                self.model_config = {
                    'name': 'NeuroQ-Micro',
                    'vocab_size': self.tokenizer.vocab_size,
                    'embed_dim': 128,
                    'num_heads': 4,
                    'num_layers': 3,
                    'max_seq_len': 256,
                    'dropout': 0.1,
                }

            # 語彙サイズをトークナイザーに合わせる
            self.model_config['vocab_size'] = self.tokenizer.vocab_size

            logger.info(f"モデル設定: {self.model_config}")

            config = NeuroQuantumConfig()
            config.vocab_size = self.model_config['vocab_size']
            config.embed_dim = self.model_config['embed_dim']
            config.num_heads = self.model_config['num_heads']
            config.num_layers = self.model_config['num_layers']
            config.max_seq_len = self.model_config.get('max_seq_len', 256)
            config.dropout = self.model_config.get('dropout', 0.1)

            self.model = NeuroQuantum(config).to(self.device)

        except ImportError:
            # フォールバック: NeuroQuantumBrain
            from neuroquantum_brain import NeuroQuantumBrain

            self.model_config = {
                'vocab_size': self.tokenizer.vocab_size,
                'embed_dim': 128,
                'num_heads': 4,
                'num_layers': 3,
                'num_neurons': 100,
            }

            self.model = NeuroQuantumBrain(
                vocab_size=self.model_config['vocab_size'],
                embed_dim=self.model_config['embed_dim'],
                num_heads=self.model_config['num_heads'],
                num_layers=self.model_config['num_layers'],
                num_neurons=self.model_config['num_neurons'],
            ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"総パラメータ数: {total_params:,} ({total_params/1e6:.1f}M)")

    def prepare_data(self, conversations: List[str]) -> List[List[int]]:
        """会話データをトークン化してシーケンスを作成"""
        logger.info("データをトークン化中...")

        all_tokens = []
        for conv in conversations:
            tokens = self.tokenizer.encode(conv, add_special=False)
            all_tokens.extend(tokens)
            all_tokens.append(self.tokenizer.eos_id)

        logger.info(f"総トークン数: {len(all_tokens):,}")

        # シーケンスを作成（オーバーラップあり）
        seq_length = self.config.seq_length
        sequences = []
        for i in range(0, len(all_tokens) - seq_length, seq_length // 2):
            sequences.append(all_tokens[i:i + seq_length])

        logger.info(f"シーケンス数: {len(sequences):,}")

        return sequences

    def train(self, sequences: List[List[int]]) -> float:
        """モデルを学習"""
        logger.info("学習開始...")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        criterion = torch.nn.CrossEntropyLoss()

        # Mixed Precision (CUDA使用時)
        use_amp = self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        batch_size = self.config.batch_size
        epochs = self.config.epochs
        best_loss = float('inf')

        total_batches = len(sequences) // batch_size
        logger.info(f"エポックあたりバッチ数: {total_batches:,}")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0

            # シャッフル
            random.shuffle(sequences)

            for i in range(0, len(sequences) - batch_size, batch_size):
                batch = sequences[i:i + batch_size]
                batch_tensor = torch.tensor(batch, device=self.device)

                input_ids = batch_tensor[:, :-1]
                target_ids = batch_tensor[:, 1:]

                optimizer.zero_grad()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(input_ids)
                        loss = criterion(
                            logits.reshape(-1, self.model_config['vocab_size']),
                            target_ids.reshape(-1)
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self.model(input_ids)
                    loss = criterion(
                        logits.reshape(-1, self.model_config['vocab_size']),
                        target_ids.reshape(-1)
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # 進捗表示
                if batch_count % max(10, total_batches // 10) == 0:
                    progress = batch_count / total_batches * 100
                    avg_loss = total_loss / batch_count
                    logger.info(f"Epoch {epoch+1}/{epochs} - {progress:.0f}% - Loss: {avg_loss:.4f}")

            avg_loss = total_loss / max(batch_count, 1)
            logger.info(f"Epoch {epoch+1}/{epochs} 完了: Loss={avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss

        logger.info(f"学習完了! ベストロス: {best_loss:.4f}")
        return best_loss

    def save_checkpoint(self, conversations_count: int) -> str:
        """チェックポイントを保存"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        checkpoint_name = f"neuroq_oasst_{self.config.model_size}_checkpoint.pt"
        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_name)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.model_config,
            'tokenizer': {
                'type': 'tiktoken',
                'encoding': 'o200k_base',
                'vocab_size': self.tokenizer.vocab_size,
            },
            'training_info': {
                'dataset': f'OpenAssistant/{self.config.dataset_name}',
                'languages': self.config.languages,
                'conversations_count': conversations_count,
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
            },
            'model_size': self.config.model_size,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"チェックポイント保存: {checkpoint_path}")

        return checkpoint_path

    def test_generation(self, prompts: List[str] = None):
        """生成テスト"""
        if prompts is None:
            prompts = [
                "<USER> What is artificial intelligence? <ASSISTANT>",
                "<USER> Hello, how can you help me? <ASSISTANT>",
                "<USER> Explain quantum computing. <ASSISTANT>",
            ]

        logger.info("生成テスト...")
        self.model.eval()

        for prompt in prompts:
            try:
                input_ids = self.tokenizer.encode(prompt)
                generated = input_ids.copy()

                with torch.no_grad():
                    for _ in range(80):
                        seq_tensor = torch.tensor(
                            [generated[-256:]],
                            device=self.device
                        )
                        logits = self.model(seq_tensor)

                        probs = torch.softmax(logits[0, -1, :] / 0.8, dim=-1)

                        # Top-k sampling
                        top_k = 50
                        top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)))
                        top_probs = top_probs / top_probs.sum()
                        idx = torch.multinomial(top_probs, num_samples=1)
                        next_token = top_indices[idx].item()

                        if next_token == self.tokenizer.eos_id:
                            break
                        generated.append(next_token)

                output = self.tokenizer.decode(generated, skip_special=True)

                # 応答部分を抽出
                if '<ASSISTANT>' in output:
                    response = output.split('<ASSISTANT>')[-1].strip()
                else:
                    response = output

                if '<USER>' in response:
                    response = response.split('<USER>')[0].strip()

                question = prompt.replace('<USER> ', '').replace(' <ASSISTANT>', '')
                logger.info(f"Q: {question[:50]}...")
                logger.info(f"A: {response[:150]}...")

            except Exception as e:
                logger.error(f"生成エラー: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='OpenAssistant Conversations API学習スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    # 英語データで学習
    python train_openassistant_api.py --language en

    # 日本語と英語で学習
    python train_openassistant_api.py --language ja,en

    # Smallモデルで学習
    python train_openassistant_api.py --model-size small --epochs 20

    # OASST2データセットを使用
    python train_openassistant_api.py --dataset oasst2
"""
    )

    parser.add_argument(
        '--model-size', '-m',
        choices=['micro', 'small', 'large'],
        default='micro',
        help='モデルサイズ (default: micro)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=30,
        help='学習エポック数 (default: 30)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8,
        help='バッチサイズ (default: 8)'
    )
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=0.0005,
        help='学習率 (default: 0.0005)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=5000,
        help='最大サンプル数 (default: 5000)'
    )
    parser.add_argument(
        '--language', '-l',
        type=str,
        default='en',
        help='言語フィルタ（カンマ区切り、例: en,ja）(default: en)'
    )
    parser.add_argument(
        '--dataset', '-d',
        choices=['oasst1', 'oasst2'],
        default='oasst1',
        help='使用するデータセット (default: oasst1)'
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=128,
        help='シーケンス長 (default: 128)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='チェックポイント保存ディレクトリ (default: checkpoints)'
    )

    args = parser.parse_args()

    # 設定
    languages = [lang.strip() for lang in args.language.split(',')]

    config = TrainingConfig(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seq_length=args.seq_length,
        max_samples=args.max_samples,
        languages=languages,
        dataset_name=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("=" * 70)
    print("OpenAssistant Conversations API 学習スクリプト")
    print("=" * 70)
    print(f"データセット: OpenAssistant/{config.dataset_name}")
    print(f"言語: {config.languages}")
    print(f"モデル: {config.model_size.upper()}")
    print(f"エポック: {config.epochs}")
    print(f"バッチサイズ: {config.batch_size}")
    print(f"最大サンプル: {config.max_samples}")
    print("=" * 70)

    # データローダー初期化
    print("\n[1/5] データセットをロード...")
    data_loader = OpenAssistantDataLoader(config.dataset_name)
    dataset = data_loader.load_dataset()

    # 会話ツリー構築
    print("\n[2/5] 会話ツリーを構築...")
    data_loader.build_conversation_tree(dataset)

    # 会話抽出
    print("\n[3/5] 会話を抽出...")
    conversations = data_loader.extract_conversations(
        languages=config.languages,
        max_samples=config.max_samples,
        include_multiturn=True
    )

    if not conversations:
        print("会話が見つかりませんでした。言語設定を確認してください。")
        sys.exit(1)

    # サンプル表示
    print("\nサンプル会話:")
    for i, conv in enumerate(conversations[:3]):
        print(f"  {i+1}. {conv[:100]}...")

    # トレーナー初期化
    print("\n[4/5] モデルを初期化...")
    trainer = NeuroQTrainer(config)
    trainer.initialize_model()

    # データ準備
    sequences = trainer.prepare_data(conversations)

    # 学習
    print("\n[5/5] 学習開始...")
    best_loss = trainer.train(sequences)

    # チェックポイント保存
    checkpoint_path = trainer.save_checkpoint(len(conversations))

    # 生成テスト
    print("\n生成テスト:")
    trainer.test_generation()

    print("\n" + "=" * 70)
    print("完了!")
    print("=" * 70)
    print(f"チェックポイント: {checkpoint_path}")
    print(f"ベストロス: {best_loss:.4f}")
    print(f"学習データ: {len(conversations):,} 会話")
    print("\n使用方法:")
    print(f"  python chat.py --checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()
