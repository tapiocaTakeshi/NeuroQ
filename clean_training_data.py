#!/usr/bin/env python3
"""
NeuroQ Training Data Cleaner
=============================
低品質な機械翻訳データを除外し、高品質な日本語会話データのみを抽出するスクリプト

主な機能：
1. 日本語率が低いテキストを除外
2. 機械翻訳特有の不自然なパターンを検出
3. 意味不明な文字列・記号だらけのテキストを除外
4. 会話形式を正しく整形
"""

import re
import unicodedata
from pathlib import Path
from typing import List, Tuple, Optional
import json
from dataclasses import dataclass


@dataclass
class CleaningStats:
    """クリーニング統計"""
    total_conversations: int = 0
    removed_low_japanese_ratio: int = 0
    removed_bad_translation: int = 0
    removed_short: int = 0
    removed_symbols: int = 0
    kept_conversations: int = 0


def is_japanese_char(char: str) -> bool:
    """日本語文字（ひらがな、カタカナ、漢字）かどうか判定"""
    if len(char) != 1:
        return False
    name = unicodedata.name(char, '')
    return any(x in name for x in ['HIRAGANA', 'KATAKANA', 'CJK UNIFIED'])


def is_hiragana(char: str) -> bool:
    """ひらがなかどうか"""
    return '\u3040' <= char <= '\u309f'


def is_katakana(char: str) -> bool:
    """カタカナかどうか"""
    return '\u30a0' <= char <= '\u30ff'


def is_kanji(char: str) -> bool:
    """漢字かどうか"""
    return '\u4e00' <= char <= '\u9fff'


def calculate_japanese_ratio(text: str) -> float:
    """テキスト中の日本語文字の割合を計算"""
    if not text:
        return 0.0

    # 空白・改行を除外
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0

    japanese_count = sum(1 for c in chars if is_japanese_char(c))
    return japanese_count / len(chars)


def calculate_hiragana_ratio(text: str) -> float:
    """テキスト中のひらがなの割合（自然な日本語の指標）"""
    if not text:
        return 0.0

    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0

    hiragana_count = sum(1 for c in chars if is_hiragana(c))
    return hiragana_count / len(chars)


def has_bad_translation_patterns(text: str) -> bool:
    """機械翻訳特有の不自然なパターンを検出"""

    bad_patterns = [
        # 英語がそのまま残っている（固有名詞以外）
        r'\b(the|is|are|was|were|have|has|had|will|would|could|should|can|may|might)\b',
        r'\b(and|or|but|if|then|because|however|therefore|although)\b',

        # 不自然な日本語パターン
        r'することができます(?:。|！)',  # 「〜することができます」の過剰使用
        r'(?:する|した)ことを(?:意味|示)',  # 英語構文の直訳
        r'これは.{0,20}です(?:。|$)',  # 「This is X」の直訳

        # 明らかな誤訳パターン
        r'貴族(?:は|が|の).{0,20}(?:元素|ガス|気体)',  # noble gas → 貴族
        r'(?:派生物|誘導体)(?:を|が)(?:見つける|取得)',  # derivative の誤訳

        # 不自然なカタカナ語
        r'(?:モノプ[ソンド]+ニー|パーセプト[ロン]+)',  # 意味不明なカタカナ
        r'(?:センサリオ?モトラ|プレイアップ)',  # 英語の音訳ミス

        # 中国語・韓国語の混入
        r'[\u4e00-\u9fff]{10,}',  # 漢字だけ10文字以上連続（中国語の可能性）
        r'[\uac00-\ud7af]',  # ハングル

        # ロシア語・キリル文字
        r'[\u0400-\u04ff]',

        # アラビア語
        r'[\u0600-\u06ff]',
    ]

    for pattern in bad_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def has_too_many_symbols(text: str, threshold: float = 0.3) -> bool:
    """記号が多すぎるかチェック"""
    if not text:
        return True

    # 通常の記号（句読点を除く）
    symbol_pattern = r'[!@#$%^&*()_+=\[\]{}|\\<>?/~`]'
    symbols = len(re.findall(symbol_pattern, text))

    # 全角記号
    fullwidth_symbols = len(re.findall(r'[！＠＃＄％＾＆＊（）＿＋＝｛｝｜＜＞？]', text))

    total_symbols = symbols + fullwidth_symbols
    chars = len([c for c in text if not c.isspace()])

    if chars == 0:
        return True

    return (total_symbols / chars) > threshold


def is_natural_japanese(text: str) -> bool:
    """自然な日本語かどうかを総合判定"""

    # 1. 日本語率チェック（50%以上）
    jp_ratio = calculate_japanese_ratio(text)
    if jp_ratio < 0.5:
        return False

    # 2. ひらがな率チェック（10%以上 - 自然な日本語にはひらがなが必ずある）
    hiragana_ratio = calculate_hiragana_ratio(text)
    if hiragana_ratio < 0.1:
        return False

    # 3. 機械翻訳パターンチェック
    if has_bad_translation_patterns(text):
        return False

    # 4. 記号過多チェック
    if has_too_many_symbols(text):
        return False

    return True


def extract_conversation_pairs(text: str) -> List[Tuple[str, str]]:
    """テキストからUser/Assistantのペアを抽出"""
    pairs = []

    # User: ... Assistant: ... のパターンを抽出
    pattern = r'User:\s*(.+?)(?=\nAssistant:)\nAssistant:\s*(.+?)(?=\n\nUser:|\n\n\n|$)'

    matches = re.findall(pattern, text, re.DOTALL)

    for user_text, assistant_text in matches:
        user_text = user_text.strip()
        assistant_text = assistant_text.strip()

        if user_text and assistant_text:
            pairs.append((user_text, assistant_text))

    return pairs


def clean_conversation(user: str, assistant: str) -> Optional[Tuple[str, str]]:
    """会話ペアをクリーニング"""

    # 両方が自然な日本語であることを確認
    if not is_natural_japanese(user) or not is_natural_japanese(assistant):
        return None

    # 短すぎる会話を除外
    if len(user) < 5 or len(assistant) < 10:
        return None

    # テキストのクリーニング
    user = clean_text(user)
    assistant = clean_text(assistant)

    if not user or not assistant:
        return None

    return (user, assistant)


def clean_text(text: str) -> str:
    """テキストをクリーニング"""

    # 制御文字を除去
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # 連続する空白を1つに
    text = re.sub(r'[ \t]+', ' ', text)

    # 連続する改行を1つに
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 先頭と末尾の空白を除去
    text = text.strip()

    return text


def clean_oasst1_ja_file(input_path: str, output_path: str) -> CleaningStats:
    """oasst1_ja_conversations.txt をクリーニング"""

    stats = CleaningStats()
    cleaned_conversations = []

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # User/Assistant ペアを抽出（行ベース）
    current_user = None
    current_assistant = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('User:'):
            # 新しいUserメッセージ
            if current_user and current_assistant:
                # 前のペアを保存
                stats.total_conversations += 1
                result = evaluate_and_clean_pair(current_user, current_assistant, stats)
                if result:
                    cleaned_conversations.append(result)
                    stats.kept_conversations += 1

            current_user = line[5:].strip()  # "User:" を削除
            current_assistant = None

            # 複数行のUserメッセージを結合
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(('User:', 'Assistant:')):
                if lines[i].strip():
                    current_user += ' ' + lines[i].strip()
                i += 1
            continue

        elif line.startswith('Assistant:'):
            current_assistant = line[10:].strip()  # "Assistant:" を削除

            # 複数行のAssistantメッセージを結合
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(('User:', 'Assistant:')):
                if lines[i].strip():
                    current_assistant += ' ' + lines[i].strip()
                i += 1
            continue

        i += 1

    # 最後のペアを処理
    if current_user and current_assistant:
        stats.total_conversations += 1
        result = evaluate_and_clean_pair(current_user, current_assistant, stats)
        if result:
            cleaned_conversations.append(result)
            stats.kept_conversations += 1

    # 出力
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(cleaned_conversations))

    return stats


def evaluate_and_clean_pair(user: str, assistant: str, stats: CleaningStats) -> Optional[str]:
    """会話ペアを評価してクリーニング"""

    # 日本語率チェック（40%以上に緩和）
    user_jp = calculate_japanese_ratio(user)
    assistant_jp = calculate_japanese_ratio(assistant)

    if user_jp < 0.4 or assistant_jp < 0.4:
        stats.removed_low_japanese_ratio += 1
        return None

    # 重大な機械翻訳エラーのみチェック（緩和版）
    if has_severe_translation_issues(user) or has_severe_translation_issues(assistant):
        stats.removed_bad_translation += 1
        return None

    # 長さチェック
    if len(user) < 3 or len(assistant) < 5:
        stats.removed_short += 1
        return None

    # 記号過多チェック（閾値を緩和）
    if has_too_many_symbols(user, 0.4) or has_too_many_symbols(assistant, 0.4):
        stats.removed_symbols += 1
        return None

    user = clean_text(user)
    assistant = clean_text(assistant)

    return f"User: {user}\nAssistant: {assistant}"


def has_severe_translation_issues(text: str) -> bool:
    """重大な翻訳問題のみを検出（緩和版）"""

    severe_patterns = [
        # 明らかな誤訳
        r'貴族(?:は|が|の).{0,20}(?:元素|ガス|気体|化学)',  # noble gas → 貴族

        # キリル文字（ロシア語）
        r'[\u0400-\u04ff]{3,}',

        # アラビア語
        r'[\u0600-\u06ff]{3,}',

        # ハングル
        r'[\uac00-\ud7af]{3,}',

        # 中国語が主体（漢字だけが15文字以上連続）
        r'[\u4e00-\u9fff]{15,}',

        # 意味不明な音訳
        r'(?:センサリオモトラ|プレイアップゲーム)',
    ]

    for pattern in severe_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def create_high_quality_training_data(output_path: str) -> int:
    """高品質な日本語学習データを作成"""

    # 手作りの高品質会話データ
    conversations = [
        # 挨拶・基本会話
        ("こんにちは", "こんにちは！何かお手伝いできることはありますか？"),
        ("おはようございます", "おはようございます！今日も素敵な一日になりますように。"),
        ("こんばんは", "こんばんは！何かお話しましょうか？"),
        ("あなたは誰ですか？", "私はニューロQという名前のAIアシスタントです。質問に答えたり、お手伝いをしたりできます。"),
        ("何ができますか？", "私は会話を通じて質問に答えたり、情報を提供したりできます。プログラミング、科学、日常的な質問など、様々なトピックについてお話できます。"),

        # プログラミング
        ("Pythonって何？", "Pythonは読みやすく書きやすいプログラミング言語です。データ分析、ウェブ開発、AI開発など幅広い分野で使われています。"),
        ("プログラミングを始めたいです", "素晴らしいですね！まずはPythonから始めることをお勧めします。文法がシンプルで、初心者にも優しい言語です。"),
        ("変数って何ですか？", "変数はデータを保存する箱のようなものです。例えば「x = 10」と書くと、xという名前の箱に10という値が入ります。"),
        ("関数の作り方を教えて", "Pythonでは「def 関数名():」という形で関数を定義します。例えば「def hello(): print('こんにちは')」のように書きます。"),
        ("リストって何？", "リストは複数のデータをまとめて管理できる機能です。例えば「fruits = ['りんご', 'みかん', 'バナナ']」のように使います。"),

        # AI・機械学習
        ("機械学習って何？", "機械学習は、コンピュータがデータから自動的にパターンを学ぶ技術です。例えば、写真から猫と犬を区別することを学べます。"),
        ("ニューラルネットワークとは？", "ニューラルネットワークは人間の脳の仕組みを参考にした計算モデルです。入力から出力を予測することを学習できます。"),
        ("深層学習って？", "深層学習はニューラルネットワークを何層も重ねた手法です。画像認識や自然言語処理で高い性能を発揮します。"),
        ("AIと機械学習の違いは？", "AIは人工知能全般を指し、機械学習はその一分野です。機械学習はデータから学習する手法で、AIを実現する方法の一つです。"),

        # 科学
        ("量子コンピュータって何？", "量子コンピュータは量子力学の性質を利用したコンピュータです。特定の計算を従来のコンピュータより高速に行える可能性があります。"),
        ("光の速さは？", "光の速さは真空中で秒速約30万キロメートルです。これは宇宙で最も速い速度とされています。"),
        ("地球と太陽の距離は？", "地球と太陽の平均距離は約1億5000万キロメートルです。これを天文単位と呼び、1AUと表します。"),
        ("水の化学式は？", "水の化学式はH2Oです。水素原子2つと酸素原子1つで構成されています。"),

        # 日常会話
        ("今日の天気は？", "申し訳ありませんが、私はリアルタイムの天気情報を持っていません。天気予報サイトやアプリで確認してください。"),
        ("おすすめの本はありますか？", "どのようなジャンルの本をお探しですか？小説、ビジネス書、技術書など、興味のある分野を教えてください。"),
        ("料理のコツを教えて", "料理の基本は火加減と味付けです。まずは簡単なレシピから始めて、少しずつ難しい料理に挑戦していくとよいでしょう。"),
        ("運動を始めたいです", "素晴らしいですね！まずは毎日の散歩やストレッチから始めることをお勧めします。無理せず継続することが大切です。"),

        # 学習・教育
        ("効率的な勉強法は？", "効率的な勉強法として、ポモドーロテクニック（25分集中、5分休憩）がおすすめです。また、学んだことを人に説明すると理解が深まります。"),
        ("暗記のコツは？", "暗記のコツは繰り返し復習することです。忘却曲線を意識して、1日後、3日後、1週間後に復習すると効果的です。"),
        ("集中力を上げるには？", "集中力を上げるには、適度な休憩、十分な睡眠、そして集中できる環境を整えることが大切です。"),

        # 感情・共感
        ("疲れました", "お疲れさまです。無理せず、ゆっくり休んでください。何かお話を聞きましょうか？"),
        ("悲しいです", "お気持ちお察しします。よければ、何があったか話してみませんか？話すことで少し楽になることもあります。"),
        ("嬉しいことがありました！", "それは素晴らしいですね！よければ何があったか教えてください。一緒に喜びましょう！"),
        ("不安です", "不安な気持ちは誰にでもあります。具体的に何が不安ですか？話すことで整理できるかもしれません。"),
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        for user, assistant in conversations:
            f.write(f"User: {user}\nAssistant: {assistant}\n\n\n")

    return len(conversations)


def main():
    """メイン処理"""

    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"

    print("=" * 60)
    print("NeuroQ Training Data Cleaner")
    print("=" * 60)

    # 1. oasst1_ja のクリーニング
    input_file = data_dir / "oasst1_ja_conversations.txt"
    output_file = data_dir / "oasst1_ja_cleaned.txt"

    if input_file.exists():
        print(f"\n[1] Cleaning {input_file.name}...")
        stats = clean_oasst1_ja_file(str(input_file), str(output_file))

        print(f"    Total conversations: {stats.total_conversations}")
        print(f"    Removed (low Japanese ratio): {stats.removed_low_japanese_ratio}")
        print(f"    Removed (bad translation): {stats.removed_bad_translation}")
        print(f"    Removed (too short): {stats.removed_short}")
        print(f"    Removed (too many symbols): {stats.removed_symbols}")
        print(f"    Kept conversations: {stats.kept_conversations}")
        print(f"    Output: {output_file}")
    else:
        print(f"\n[1] {input_file} not found, skipping...")

    # 2. 高品質会話データの生成
    hq_file = data_dir / "high_quality_conversations.txt"
    print(f"\n[2] Creating high-quality conversation data...")
    count = create_high_quality_training_data(str(hq_file))
    print(f"    Created {count} high-quality conversations")
    print(f"    Output: {hq_file}")

    # 3. 統合データの作成
    combined_file = data_dir / "combined_clean_data.txt"
    print(f"\n[3] Creating combined clean data...")

    combined_content = []

    # 高品質データを最初に（重み付け）
    if hq_file.exists():
        with open(hq_file, 'r', encoding='utf-8') as f:
            hq_content = f.read()
        # 高品質データを3回繰り返し（重み付け）
        combined_content.extend([hq_content] * 3)

    # クリーニング済みoasst1データ
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            cleaned_content = f.read()
        combined_content.append(cleaned_content)

    with open(combined_file, 'w', encoding='utf-8') as f:
        f.write('\n\n\n'.join(combined_content))

    print(f"    Output: {combined_file}")

    print("\n" + "=" * 60)
    print("Cleaning complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the cleaned data in data/oasst1_ja_cleaned.txt")
    print("2. Train with: python train_oasst1_ja.py --data data/combined_clean_data.txt")
    print("3. Use lower temperature (0.3-0.7) during generation")


if __name__ == "__main__":
    main()
