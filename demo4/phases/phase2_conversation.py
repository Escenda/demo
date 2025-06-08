import os
import json
import random
import re
from typing import List, Dict, Tuple
from tqdm import tqdm
import requests


class Phase2ConversationDatasetGenerator:
    """
    フェーズ2: フェーズ1の語彙のみを使った会話データセット生成
    Q&A形式の対話データを自動生成
    """
    
    def __init__(self, phase1_dir: str = "data/phase1", output_dir: str = "data/phase2"):
        self.phase1_dir = phase1_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # フェーズ1の語彙を読み込む
        self.phase1_vocabulary = set()
        self.phase1_knowledge = []
        self._load_phase1_data()
        
        # ローカルLLM設定
        self.llm_endpoint = "http://localhost:11434/api/generate"
        self.llm_model = "gemma3:4b-it-qat"
        
        # 会話テンプレート
        self.conversation_templates = self._init_conversation_templates()
    
    def _load_phase1_data(self):
        """フェーズ1のデータと語彙を読み込む"""
        phase1_file = os.path.join(self.phase1_dir, "knowledge_dataset.jsonl")
        
        if not os.path.exists(phase1_file):
            print(f"Warning: Phase 1 data not found at {phase1_file}")
            return
        
        with open(phase1_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                self.phase1_knowledge.append(entry)
                
                # 語彙を抽出（簡易的な実装）
                text = entry["full_explanation"] + " ".join(entry.get("definitions", []))
                words = self._extract_words(text)
                self.phase1_vocabulary.update(words)
        
        print(f"Loaded {len(self.phase1_knowledge)} knowledge entries")
        print(f"Phase 1 vocabulary size: {len(self.phase1_vocabulary)}")
    
    def _extract_words(self, text: str) -> List[str]:
        """テキストから単語を抽出（日本語対応）"""
        # 簡易的な実装：句読点で分割して単語を取得
        # 実際のプロダクションではMeCab等を使用
        words = []
        
        # 基本的な単語境界で分割
        text = re.sub(r'[。、！？\n]', ' ', text)
        tokens = text.split()
        
        for token in tokens:
            # 英数字、ひらがな、カタカナ、漢字を含む部分を抽出
            matches = re.findall(r'[\w\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+', token)
            words.extend(matches)
        
        return words
    
    def _init_conversation_templates(self) -> List[Dict]:
        """会話テンプレートを初期化"""
        return [
            {
                "type": "definition",
                "user_template": "{item}とは何ですか？",
                "system_template": "{item}は、{explanation}です。"
            },
            {
                "type": "explanation",
                "user_template": "{item}について教えてください。",
                "system_template": "{item}について説明します。{explanation}"
            },
            {
                "type": "characteristic",
                "user_template": "{item}の特徴は何ですか？",
                "system_template": "{item}の主な特徴は、{features}です。"
            },
            {
                "type": "comparison",
                "user_template": "{item1}と{item2}の違いは何ですか？",
                "system_template": "{item1}と{item2}の違いは、{difference}です。"
            },
            {
                "type": "example",
                "user_template": "{item}の例を教えてください。",
                "system_template": "{item}の例として、{examples}が挙げられます。"
            },
            {
                "type": "usage",
                "user_template": "{item}はどのように使われますか？",
                "system_template": "{item}は、{usage}のように使われます。"
            },
            {
                "type": "history",
                "user_template": "{item}の歴史について教えてください。",
                "system_template": "{item}の歴史は、{history}です。"
            },
            {
                "type": "importance",
                "user_template": "なぜ{item}は重要なのですか？",
                "system_template": "{item}が重要な理由は、{reason}です。"
            }
        ]
    
    def generate_qa_pair(self, knowledge_entry: Dict) -> List[Tuple[str, str]]:
        """知識エントリから複数のQ&Aペアを生成"""
        qa_pairs = []
        title = knowledge_entry["title"]
        explanation = knowledge_entry["full_explanation"]
        
        # 1. 基本的な定義Q&A
        template = self.conversation_templates[0]
        q = template["user_template"].format(item=title)
        a = template["system_template"].format(item=title, explanation=self._summarize_text(explanation, 100))
        qa_pairs.append((q, a))
        
        # 2. その他のテンプレートからランダムに選択
        selected_templates = random.sample(self.conversation_templates[1:], min(3, len(self.conversation_templates)-1))
        
        for template in selected_templates:
            try:
                if template["type"] == "comparison" and len(self.phase1_knowledge) > 1:
                    # 比較質問の場合、別のアイテムを選択
                    other_entry = random.choice([e for e in self.phase1_knowledge if e["title"] != title])
                    q = template["user_template"].format(item1=title, item2=other_entry["title"])
                    a = self._generate_comparison_answer(knowledge_entry, other_entry)
                elif template["type"] in ["characteristic", "example", "usage", "history", "importance"]:
                    q = template["user_template"].format(item=title)
                    a = self._generate_contextual_answer(template["type"], knowledge_entry)
                else:
                    q = template["user_template"].format(item=title)
                    a = template["system_template"].format(
                        item=title,
                        explanation=self._summarize_text(explanation, 100)
                    )
                
                # フェーズ1の語彙制限をチェック
                if self._check_vocabulary_constraint(q + " " + a):
                    qa_pairs.append((q, a))
                    
            except Exception as e:
                continue
        
        return qa_pairs
    
    def _summarize_text(self, text: str, max_length: int) -> str:
        """テキストを要約（簡易版）"""
        sentences = re.split(r'[。．]', text)
        summary = ""
        
        for sent in sentences:
            if len(summary) + len(sent) <= max_length:
                summary += sent + "。"
            else:
                break
        
        return summary.strip()
    
    def _generate_comparison_answer(self, entry1: Dict, entry2: Dict) -> str:
        """比較回答を生成"""
        template = random.choice([
            "{item1}は{feature1}という特徴がありますが、{item2}は{feature2}という点で異なります。",
            "{item1}と{item2}の主な違いは、{difference}という点にあります。",
            "{item1}は{category1}に属しますが、{item2}は{category2}に分類されます。"
        ])
        
        return template.format(
            item1=entry1["title"],
            item2=entry2["title"],
            feature1=self._extract_feature(entry1["full_explanation"]),
            feature2=self._extract_feature(entry2["full_explanation"]),
            difference="それぞれの特性",
            category1=entry1.get("category", "その他"),
            category2=entry2.get("category", "その他")
        )
    
    def _extract_feature(self, text: str) -> str:
        """テキストから特徴を抽出"""
        # 「〜という」「〜である」などのパターンを探す
        patterns = [
            r'(\w+という\w+)',
            r'(\w+である)',
            r'(\w+として)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        
        # フォールバック
        return "独自の特徴"
    
    def _generate_contextual_answer(self, answer_type: str, knowledge_entry: Dict) -> str:
        """文脈に応じた回答を生成"""
        title = knowledge_entry["title"]
        explanation = knowledge_entry["full_explanation"]
        
        if answer_type == "characteristic":
            features = self._extract_features_list(explanation)
            return f"{title}の主な特徴は、{', '.join(features[:3])}などです。"
        
        elif answer_type == "example":
            # 簡易的な例の生成
            category = knowledge_entry.get("category", "その他")
            return f"{title}の例として、一般的な{category}分野での応用が挙げられます。"
        
        elif answer_type == "usage":
            return f"{title}は、日常生活や専門分野において様々な形で活用されています。"
        
        elif answer_type == "history":
            return f"{title}の歴史は古く、長い年月をかけて発展してきました。"
        
        elif answer_type == "importance":
            return f"{title}が重要な理由は、私たちの生活や社会に大きな影響を与えているからです。"
        
        return f"{title}について、{self._summarize_text(explanation, 100)}"
    
    def _extract_features_list(self, text: str) -> List[str]:
        """テキストから特徴リストを抽出"""
        features = []
        
        # 「〜性」「〜力」などの特徴を表す語を探す
        feature_patterns = [
            r'(\w+性)',
            r'(\w+力)',
            r'(\w+的)',
        ]
        
        for pattern in feature_patterns:
            matches = re.findall(pattern, text)
            features.extend(matches[:2])  # 各パターンから最大2つ
        
        if not features:
            features = ["専門性", "実用性", "汎用性"]  # デフォルト
        
        return features
    
    def _check_vocabulary_constraint(self, text: str) -> bool:
        """フェーズ1の語彙制限をチェック"""
        # 簡易実装：主要な単語がフェーズ1語彙に含まれているかチェック
        words = self._extract_words(text)
        
        # 全単語の70%以上がフェーズ1語彙に含まれていればOK
        if not words:
            return True
            
        known_words = sum(1 for w in words if w in self.phase1_vocabulary or len(w) <= 2)
        return known_words / len(words) >= 0.7
    
    def generate_conversation_with_llm(self, knowledge_entry: Dict) -> List[Dict]:
        """ローカルLLMを使用してより自然な会話を生成"""
        conversations = []
        title = knowledge_entry["title"]
        
        prompt = f"""以下の知識に基づいて、自然な日本語の会話を3つ生成してください。
知識: {knowledge_entry['full_explanation'][:500]}

要件:
- ユーザーとシステムの対話形式
- 既存の知識のみを使用（新しい情報は追加しない）
- 自然で丁寧な日本語

形式:
ユーザー: [質問]
システム: [回答]

会話:"""
        
        try:
            response = requests.post(
                self.llm_endpoint,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                # 生成されたテキストから会話を抽出
                conversations.extend(self._parse_generated_conversations(generated_text))
                
        except Exception as e:
            print(f"LLM generation error: {e}")
        
        return conversations
    
    def _parse_generated_conversations(self, text: str) -> List[Dict]:
        """生成されたテキストから会話を解析"""
        conversations = []
        
        # ユーザー/システムのパターンで分割
        pattern = r'ユーザー[:：](.*?)システム[:：](.*?)(?=ユーザー[:：]|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for user_text, system_text in matches:
            conversations.append({
                "user": user_text.strip(),
                "system": system_text.strip()
            })
        
        return conversations
    
    def process_phase2_dataset(self):
        """フェーズ2データセット生成のメインプロセス"""
        output_file = os.path.join(self.output_dir, "conversation_dataset.jsonl")
        
        all_conversations = []
        
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in tqdm(self.phase1_knowledge, desc="Generating conversations"):
                # 1. テンプレートベースのQ&A生成
                qa_pairs = self.generate_qa_pair(entry)
                
                for i, (question, answer) in enumerate(qa_pairs):
                    conversation = {
                        "id": f"{entry['title']}_qa_{i}",
                        "source_knowledge": entry["title"],
                        "conversation": [
                            {"role": "user", "content": question},
                            {"role": "system", "content": answer}
                        ],
                        "type": "template_based"
                    }
                    f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
                    all_conversations.append(conversation)
                
                # 2. LLMベースの会話生成（オプション）
                if random.random() < 0.3:  # 30%の確率でLLM生成を試みる
                    llm_conversations = self.generate_conversation_with_llm(entry)
                    for i, conv in enumerate(llm_conversations):
                        conversation = {
                            "id": f"{entry['title']}_llm_{i}",
                            "source_knowledge": entry["title"],
                            "conversation": [
                                {"role": "user", "content": conv["user"]},
                                {"role": "system", "content": conv["system"]}
                            ],
                            "type": "llm_generated"
                        }
                        f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
                        all_conversations.append(conversation)
        
        print(f"\nGenerated {len(all_conversations)} conversations")
        return all_conversations
    
    def create_training_sequences(self, tokenizer, max_length: int = 1024):
        """会話データから学習用シーケンスを作成"""
        input_file = os.path.join(self.output_dir, "conversation_dataset.jsonl")
        output_file = os.path.join(self.output_dir, "training_sequences.pt")
        
        sequences = []
        
        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Creating training sequences"):
                entry = json.loads(line)
                
                # 会話を一つのテキストに結合
                conversation_text = ""
                for turn in entry["conversation"]:
                    if turn["role"] == "user":
                        conversation_text += f"ユーザー: {turn['content']}\n"
                    else:
                        conversation_text += f"システム: {turn['content']}\n"
                
                # トークナイズ
                tokens = tokenizer.encode(conversation_text, add_special_tokens=True)
                
                # 固定長に調整
                if len(tokens) <= max_length:
                    # パディング
                    tokens.extend([tokenizer.pad_token_id] * (max_length - len(tokens)))
                    sequences.append(tokens)
                else:
                    # 長すぎる場合は分割
                    for i in range(0, len(tokens) - max_length + 1, max_length // 2):
                        seq = tokens[i:i + max_length]
                        sequences.append(seq)
        
        # PyTorchテンソルとして保存
        import torch
        sequences_tensor = torch.tensor(sequences, dtype=torch.long)
        torch.save(sequences_tensor, output_file)
        
        print(f"Created {len(sequences)} training sequences")
        return sequences_tensor
    
    def generate_phase2_dataset(self):
        """フェーズ2データセット生成のメインメソッド"""
        print("\n=== Phase 2: Generating Conversation Dataset ===")
        
        if not self.phase1_knowledge:
            print("Error: No Phase 1 data found. Please run Phase 1 first.")
            return
        
        # 会話データセットを生成
        conversations = self.process_phase2_dataset()
        
        # 統計情報を表示
        self.show_statistics()
        
        print("Phase 2 dataset generation completed!")
    
    def show_statistics(self):
        """生成したデータセットの統計情報を表示"""
        input_file = os.path.join(self.output_dir, "conversation_dataset.jsonl")
        
        stats = {
            "total_conversations": 0,
            "template_based": 0,
            "llm_generated": 0,
            "avg_conversation_length": 0
        }
        
        total_length = 0
        
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                stats["total_conversations"] += 1
                stats[entry["type"]] = stats.get(entry["type"], 0) + 1
                
                # 会話の長さを計算
                conv_length = sum(len(turn["content"]) for turn in entry["conversation"])
                total_length += conv_length
        
        stats["avg_conversation_length"] = total_length / max(stats["total_conversations"], 1)
        
        print("\n=== Dataset Statistics ===")
        print(f"Total conversations: {stats['total_conversations']}")
        print(f"Template-based: {stats.get('template_based', 0)}")
        print(f"LLM-generated: {stats.get('llm_generated', 0)}")
        print(f"Average conversation length: {stats['avg_conversation_length']:.0f} characters")


if __name__ == "__main__":
    # テスト実行
    generator = Phase2ConversationDatasetGenerator()
    generator.generate_phase2_dataset()