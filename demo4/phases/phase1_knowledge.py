import os
import json
import re
from typing import List, Dict, Generator
from datasets import load_dataset
import requests
from tqdm import tqdm


class Phase1KnowledgeDatasetGenerator:
    """
    フェーズ1: 森羅万象の知識説明データセット生成
    WikipediaやCC-100などから百科事典的な説明文を抽出・生成
    """
    
    def __init__(self, output_dir: str = "data/phase1", max_samples: int = None):
        self.output_dir = output_dir
        self.max_samples = max_samples
        os.makedirs(output_dir, exist_ok=True)
        
        # ローカルLLM設定（Ollama等を想定）
        self.llm_endpoint = "http://localhost:11434/api/generate"
        self.llm_model = "gemma3:4b-it-qat"  # または他の日本語対応モデル
        
    def download_wikipedia_ja(self) -> Generator[Dict, None, None]:
        """
        日本語Wikipediaデータセットをダウンロードしてストリーミング
        """
        print("Loading Japanese Wikipedia dataset...")
        dataset = load_dataset("graelo/wikipedia", "20230901.ja", split="train[:175]")
        print(f"Loaded {len(dataset)} articles")
        
        count = 0
        for article in dataset:
            if self.max_samples and count >= self.max_samples:
                break
                
            # 記事本文を抽出
            text = article["text"]
            title = article["title"]
            
            # 短すぎる記事はスキップ
            if len(text) < 100:
                continue
                
            yield {
                "title": title,
                "text": text,
                "source": "wikipedia"
            }
            count += 1
    
    def extract_definition_sentences(self, text: str, title: str) -> List[str]:
        """
        記事から定義文や説明文を抽出
        """
        sentences = []
        
        # 文を句点で分割
        raw_sentences = re.split(r'[。．]', text)
        
        for i, sent in enumerate(raw_sentences[:10]):  # 最初の10文まで
            sent = sent.strip()
            if not sent:
                continue
                
            # 定義文のパターンを検出
            if any(pattern in sent for pattern in ['とは', 'は、', 'である', 'です']):
                sentences.append(sent + '。')
            
            # タイトルが含まれる文も重要
            elif title in sent:
                sentences.append(sent + '。')
        
        return sentences
    
    def generate_enhanced_explanation(self, title: str, original_text: str) -> str:
        """
        ローカルLLMを使用して、より詳細な説明文を生成
        """
        prompt = f"""以下の項目について、百科事典的な詳細な説明文を生成してください。
項目: {title}
参考情報: {original_text[:500]}

要件:
- 客観的で中立的な説明
- 0から100まで完全に理解できる詳細さ
- 専門用語は適切に解説
- 日本語で記述

説明文:"""
        
        try:
            response = requests.post(
                self.llm_endpoint,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_ctx": 1000
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
        except Exception as e:
            print(f"LLM generation error: {e}")
        
        # フォールバック: 元のテキストを整形して返す
        return self._format_explanation(title, original_text)
    
    def _format_explanation(self, title: str, text: str) -> str:
        """
        元のテキストを整形して説明文にする（フォールバック用）
        """
        # 最初の段落を抽出
        paragraphs = text.split('\n\n')
        first_para = paragraphs[0] if paragraphs else text[:500]
        
        # 定義文スタイルに整形
        if not first_para.startswith(title):
            formatted = f"{title}は、{first_para}"
        else:
            formatted = first_para
            
        # 文末を整える
        if not formatted.endswith('。'):
            formatted += '。'
            
        return formatted

    def _load_existing_ids(self, output_file: str) -> set:
        """
        既存のknowledge_dataset.jsonlからIDのセットを取得
        """
        existing_ids = set()
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if "id" in entry:
                            existing_ids.add(entry["id"])
                    except Exception:
                        continue
        return existing_ids

    def process_articles(self):
        """
        記事を処理してフェーズ1データセットを生成
        既に生成済みのIDがあればスキップする
        """
        output_file = os.path.join(self.output_dir, "knowledge_dataset.jsonl")
        # 既存IDを取得
        existing_ids = self._load_existing_ids(output_file)

        # 追記モードでファイルを開く
        with open(output_file, "a", encoding="utf-8") as f:
            for article in tqdm(self.download_wikipedia_ja(), desc="Processing articles"):
                title = article["title"]
                text = article["text"]
                entry_id = f"wiki_{title}"

                # 既存IDならスキップ
                if entry_id in existing_ids:
                    continue

                # 1. 定義文を抽出
                definitions = self.extract_definition_sentences(text, title)
                
                # 2. 拡張説明文を生成（ローカルLLMまたはフォールバック）
                enhanced_explanation = self.generate_enhanced_explanation(title, text)
                
                # 3. データセットエントリを作成
                entry = {
                    "id": entry_id,
                    "title": title,
                    "definitions": definitions,
                    "full_explanation": enhanced_explanation,
                    "source": "wikipedia",
                    "category": self._categorize_article(title, text)
                }
                
                # 4. 保存
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                existing_ids.add(entry_id)  # 追加したIDもセットに加える
    
    def _categorize_article(self, title: str, text: str) -> str:
        """
        記事をカテゴリ分類（簡易版）
        """
        categories = {
            "科学": ["科学", "物理", "化学", "生物", "数学"],
            "歴史": ["歴史", "時代", "戦争", "王朝"],
            "地理": ["地理", "国", "都市", "地域"],
            "人物": ["人物", "生誕", "死去", "活動"],
            "技術": ["技術", "コンピュータ", "工学", "発明"],
            "文化": ["文化", "芸術", "文学", "音楽"],
            "社会": ["社会", "政治", "経済", "法律"]
        }
        
        text_lower = (title + text[:500]).lower()
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return "その他"
    
    def create_training_sequences(self, tokenizer, max_length: int = 1024):
        """
        トークナイズして学習用シーケンスを作成
        """
        input_file = os.path.join(self.output_dir, "knowledge_dataset.jsonl")
        output_file = os.path.join(self.output_dir, "training_sequences.pt")
        
        sequences = []
        
        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Creating training sequences"):
                entry = json.loads(line)
                
                # フル説明文をトークナイズ
                text = entry["full_explanation"]
                tokens = tokenizer.encode(text, add_special_tokens=True)
                
                # 固定長シーケンスに分割
                for i in range(0, len(tokens) - max_length + 1, max_length // 2):
                    seq = tokens[i:i + max_length]
                    if len(seq) == max_length:
                        sequences.append(seq)
        
        # PyTorchテンソルとして保存
        import torch
        sequences_tensor = torch.tensor(sequences, dtype=torch.long)
        torch.save(sequences_tensor, output_file)
        
        print(f"Created {len(sequences)} training sequences")
        return sequences_tensor
    
    def generate_phase1_dataset(self):
        """
        フェーズ1データセット生成のメインメソッド
        """
        print("=== Phase 1: Generating Knowledge Explanation Dataset ===")
        
        # 1. Wikipedia記事を処理
        self.process_articles()
        
        # 2. 統計情報を表示
        self.show_statistics()
        
        print("Phase 1 dataset generation completed!")
    
    def show_statistics(self):
        """
        生成したデータセットの統計情報を表示
        """
        input_file = os.path.join(self.output_dir, "knowledge_dataset.jsonl")
        
        stats = {
            "total_entries": 0,
            "categories": {},
            "avg_explanation_length": 0
        }
        
        total_length = 0
        
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                stats["total_entries"] += 1
                
                category = entry["category"]
                stats["categories"][category] = stats["categories"].get(category, 0) + 1
                
                total_length += len(entry["full_explanation"])
        
        stats["avg_explanation_length"] = total_length / max(stats["total_entries"], 1)
        
        print("\n=== Dataset Statistics ===")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Average explanation length: {stats['avg_explanation_length']:.0f} characters")
        print("\nCategories:")
        for category, count in sorted(stats["categories"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count}")


if __name__ == "__main__":
    # テスト実行
    generator = Phase1KnowledgeDatasetGenerator(max_samples=1000)
    generator.generate_phase1_dataset()