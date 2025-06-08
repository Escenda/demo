import os
import json
import random
import string
from typing import List, Dict, Tuple
from tqdm import tqdm
import requests
from datetime import datetime


class Phase4MetaReasoningDatasetGenerator:
    """
    フェーズ4: 未学習情報に対するメタ推論データセット生成
    未知の概念に対して推測・学習しようとする思考過程を生成
    """
    
    def __init__(self, phase1_dir: str = "data/phase1", output_dir: str = "data/phase4"):
        self.phase1_dir = phase1_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # フェーズ1の既知情報を読み込む
        self.known_concepts = set()
        self.known_categories = {}
        self._load_phase1_data()
        
        # ローカルLLM設定
        self.llm_endpoint = "http://localhost:11434/api/generate"
        self.llm_model = "gemma3:4b-it-qat"
        
        # メタ推論パターン
        self.meta_reasoning_patterns = self._init_meta_reasoning_patterns()
        
        # 未知概念のカテゴリ
        self.unknown_categories = [
            "架空技術", "未来概念", "専門用語", "新造語", 
            "仮想概念", "理論的構造", "抽象概念"
        ]
    
    def _load_phase1_data(self):
        """フェーズ1の既知情報を読み込む"""
        phase1_file = os.path.join(self.phase1_dir, "knowledge_dataset.jsonl")
        
        if not os.path.exists(phase1_file):
            print(f"Warning: Phase 1 data not found at {phase1_file}")
            return
        
        with open(phase1_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                self.known_concepts.add(entry["title"])
                category = entry.get("category", "その他")
                if category not in self.known_categories:
                    self.known_categories[category] = []
                self.known_categories[category].append(entry["title"])
        
        print(f"Loaded {len(self.known_concepts)} known concepts")
        print(f"Categories: {list(self.known_categories.keys())}")
    
    def _init_meta_reasoning_patterns(self) -> List[Dict]:
        """メタ推論パターンを初期化"""
        return [
            {
                "type": "recognition_failure",
                "template": "私は「{unknown_term}」という言葉を初めて聞いた。{unknown_term}について私はまだ何も学習していない。しかし、この言葉の構造を分析すると、{analysis}のように思える。{related_concept}という既知の概念と関連があるかもしれない。確かなことは言えないが、{hypothesis}という推測ができる。正確に理解するためには、さらなる情報が必要だ。"
            },
            {
                "type": "learning_intention",
                "template": "新しい概念「{unknown_term}」を学ぼうとしている。現時点では{unknown_term}の正確な定義を知らない。まず、語源や構成要素から推測してみる。{etymology_guess}という可能性がある。もし私の推測が正しければ、{implication}ということになる。今は仮説段階だが、今後この概念について深く学んでいきたい。"
            },
            {
                "type": "analogical_reasoning",
                "template": "「{unknown_term}」という未知の用語に遭遇した。私の知識ベースにはこの情報が含まれていない。しかし、類似した概念として{similar_concept}を知っている。もし{unknown_term}が{similar_concept}と同じカテゴリに属するなら、{inference}という特徴を持つ可能性がある。ただし、これは推測に過ぎず、実際とは異なるかもしれない。"
            },
            {
                "type": "knowledge_gap_acknowledgment",
                "template": "私は今、{unknown_term}について考えようとしているが、残念ながらこれについての知識がない。{unknown_term}という言葉から連想されるのは{association}だが、これが正しいかどうかは分からない。私の現在の理解の限界を認識し、この知識の空白を埋めるために学習を続ける必要がある。"
            },
            {
                "type": "hypothesis_formation",
                "template": "未知の概念「{unknown_term}」について仮説を立ててみる。名称から推測すると、{category_guess}に関連する概念かもしれない。もしそうだとすれば、{property_guess}という性質を持つ可能性がある。この仮説を検証するには、{verification_method}が必要だろう。現段階では推測の域を出ないが、思考の出発点としては有用だ。"
            },
            {
                "type": "metacognitive_reflection",
                "template": "「{unknown_term}」について理解しようと試みているが、私の知識には限界がある。興味深いことに、知らないことを知っているという認識自体が重要だ。{unknown_term}は{context_guess}という文脈で使われる可能性がある。既知の{known_concept}と比較することで、部分的な理解が得られるかもしれない。完全な理解には至らないが、このような推論プロセス自体に価値がある。"
            },
            {
                "type": "exploratory_thinking",
                "template": "新しく出会った「{unknown_term}」という概念を探索的に考えてみる。直接的な知識はないが、{breakdown}という要素に分解できるかもしれない。それぞれの要素から、{synthesis}という全体像が浮かび上がる。もちろん、これは暫定的な理解であり、実際の{unknown_term}とは異なる可能性が高い。しかし、未知に対するこのようなアプローチは学習の第一歩となる。"
            },
            {
                "type": "uncertainty_expression",
                "template": "「{unknown_term}」という用語について、確実なことは何も言えない。私の知識データベースにこの情報は存在しない。推測するならば、{wild_guess}という可能性があるが、確信度は非常に低い。むしろ重要なのは、この不確実性を認識し、{learning_strategy}という方法で知識を獲得していくことだ。未知との遭遇は、学習の機会でもある。"
            }
        ]
    
    def generate_unknown_terms(self, count: int = 1000) -> List[Dict]:
        """未知の用語を生成"""
        unknown_terms = []
        
        # 1. 実在しそうな専門用語風の造語
        prefixes = ["量子", "ニューロ", "サイバー", "バイオ", "ナノ", "メタ", "ハイパー", "マルチ", "クロス", "インター"]
        middles = ["フラクタル", "シナジー", "ダイナミクス", "トポロジー", "モジュール", "インターフェース", "アルゴリズム", "プロトコル"]
        suffixes = ["システム", "理論", "構造", "機構", "ネットワーク", "プロセス", "メカニズム", "フレームワーク"]
        
        # 2. カタカナ新語
        katakana_parts = ["フラックス", "コンバージェンス", "シンギュラリティ", "パラダイム", "イノベーション", "ディスラプション"]
        
        # 3. 英数字混合語
        alpha_numeric = ["X", "Z", "Q", "2.0", "3D", "4K", "5G", "AI", "VR", "AR"]
        
        used_terms = set()
        
        while len(unknown_terms) < count:
            # タイプをランダムに選択
            term_type = random.choice(["compound", "katakana", "alphanumeric", "abstract"])
            
            if term_type == "compound":
                # 複合語を生成
                term = random.choice(prefixes) + random.choice(middles) + random.choice(suffixes)
            elif term_type == "katakana":
                # カタカナ語を生成
                parts = random.sample(katakana_parts, 2)
                term = "".join(parts)
            elif term_type == "alphanumeric":
                # 英数字混合語を生成
                term = random.choice(alpha_numeric) + random.choice(middles)
            else:
                # 抽象的な造語を生成
                term = self._generate_abstract_term()
            
            # 既知の概念と重複しないことを確認
            if term not in self.known_concepts and term not in used_terms:
                used_terms.add(term)
                unknown_terms.append({
                    "term": term,
                    "category": random.choice(self.unknown_categories),
                    "type": term_type
                })
        
        return unknown_terms
    
    def _generate_abstract_term(self) -> str:
        """抽象的な造語を生成"""
        # 音韻的にそれっぽい造語を生成
        patterns = [
            "CVC" + "CVC",  # 子音-母音-子音の繰り返し
            "CV" + "CV" + "CV",  # 子音-母音の繰り返し
            "VC" + "CVC",  # 母音始まり
        ]
        
        consonants = "ksthmyrwgnzpbdfjl"
        vowels = "aiueo"
        
        pattern = random.choice(patterns)
        term = ""
        
        for char in pattern:
            if char == "C":
                term += random.choice(consonants)
            elif char == "V":
                term += random.choice(vowels)
        
        # カタカナ風の表記に変換
        katakana_map = {
            'a': 'ア', 'i': 'イ', 'u': 'ウ', 'e': 'エ', 'o': 'オ',
            'ka': 'カ', 'ki': 'キ', 'ku': 'ク', 'ke': 'ケ', 'ko': 'コ',
            'sa': 'サ', 'si': 'シ', 'su': 'ス', 'se': 'セ', 'so': 'ソ',
            'ta': 'タ', 'ti': 'チ', 'tu': 'ツ', 'te': 'テ', 'to': 'ト',
            'na': 'ナ', 'ni': 'ニ', 'nu': 'ヌ', 'ne': 'ネ', 'no': 'ノ',
            'ha': 'ハ', 'hi': 'ヒ', 'hu': 'フ', 'he': 'ヘ', 'ho': 'ホ',
            'ma': 'マ', 'mi': 'ミ', 'mu': 'ム', 'me': 'メ', 'mo': 'モ',
            'ya': 'ヤ', 'yu': 'ユ', 'yo': 'ヨ',
            'ra': 'ラ', 'ri': 'リ', 'ru': 'ル', 're': 'レ', 'ro': 'ロ',
            'wa': 'ワ', 'wo': 'ヲ', 'n': 'ン'
        }
        
        # 簡易的なカタカナ変換
        katakana_term = ""
        i = 0
        while i < len(term):
            if i + 1 < len(term) and term[i:i+2] in katakana_map:
                katakana_term += katakana_map[term[i:i+2]]
                i += 2
            elif term[i] in katakana_map:
                katakana_term += katakana_map[term[i]]
                i += 1
            else:
                i += 1
        
        return katakana_term if katakana_term else term.upper()
    
    def generate_meta_reasoning(self, unknown_term_data: Dict) -> List[Dict]:
        """未知の用語に対するメタ推論を生成"""
        meta_reasonings = []
        unknown_term = unknown_term_data["term"]
        category = unknown_term_data["category"]
        
        # 複数のパターンを選択
        selected_patterns = random.sample(self.meta_reasoning_patterns, min(3, len(self.meta_reasoning_patterns)))
        
        for pattern in selected_patterns:
            try:
                # パターンに応じた推論を生成
                if pattern["type"] == "recognition_failure":
                    reasoning = self._generate_recognition_failure(unknown_term, category)
                elif pattern["type"] == "learning_intention":
                    reasoning = self._generate_learning_intention(unknown_term, category)
                elif pattern["type"] == "analogical_reasoning":
                    reasoning = self._generate_analogical_reasoning(unknown_term, category)
                elif pattern["type"] == "knowledge_gap_acknowledgment":
                    reasoning = self._generate_knowledge_gap(unknown_term, category)
                elif pattern["type"] == "hypothesis_formation":
                    reasoning = self._generate_hypothesis(unknown_term, category)
                elif pattern["type"] == "metacognitive_reflection":
                    reasoning = self._generate_metacognitive(unknown_term, category)
                elif pattern["type"] == "exploratory_thinking":
                    reasoning = self._generate_exploratory(unknown_term, category)
                elif pattern["type"] == "uncertainty_expression":
                    reasoning = self._generate_uncertainty(unknown_term, category)
                else:
                    continue
                
                if reasoning:
                    meta_reasonings.append({
                        "type": pattern["type"],
                        "content": reasoning,
                        "unknown_term": unknown_term,
                        "category": category
                    })
                    
            except Exception as e:
                continue
        
        return meta_reasonings
    
    def _generate_recognition_failure(self, term: str, category: str) -> str:
        """認識失敗パターンの推論を生成"""
        # 語の構造を分析
        if any(prefix in term for prefix in ["量子", "ニューロ", "サイバー", "バイオ"]):
            analysis = "科学技術に関連する専門用語"
        elif any(char.isdigit() for char in term):
            analysis = "バージョンや世代を示す技術用語"
        else:
            analysis = "何らかの理論的概念"
        
        # 関連する既知概念を選択
        related_concepts = []
        for cat, concepts in self.known_categories.items():
            if cat in ["科学", "技術"]:
                related_concepts.extend(concepts[:3])
        
        related_concept = random.choice(related_concepts) if related_concepts else "一般的な概念"
        
        # 推測を生成
        hypotheses = [
            "新しい研究分野を指す",
            "既存技術の発展形である",
            "理論的なフレームワークを表す"
        ]
        hypothesis = random.choice(hypotheses)
        
        template = self.meta_reasoning_patterns[0]["template"]
        return template.format(
            unknown_term=term,
            analysis=analysis,
            related_concept=related_concept,
            hypothesis=hypothesis
        )
    
    def _generate_learning_intention(self, term: str, category: str) -> str:
        """学習意図パターンの推論を生成"""
        # 語源の推測
        if "フラクタル" in term:
            etymology_guess = "幾何学的な自己相似性を持つ構造に関連"
        elif "シナジー" in term:
            etymology_guess = "相乗効果や協調作用を示す"
        elif "トポロジー" in term:
            etymology_guess = "位相幾何学的な性質に関連"
        else:
            etymology_guess = "複合的な概念を表す新しい用語"
        
        # 含意の推測
        implications = [
            "従来の方法論を超えた新しいアプローチ",
            "複数の分野を統合した学際的概念",
            "次世代の技術パラダイム"
        ]
        implication = random.choice(implications)
        
        template = self.meta_reasoning_patterns[1]["template"]
        return template.format(
            unknown_term=term,
            etymology_guess=etymology_guess,
            implication=implication
        )
    
    def _generate_analogical_reasoning(self, term: str, category: str) -> str:
        """類推的推論パターンを生成"""
        # カテゴリに基づいて類似概念を選択
        category_map = {
            "架空技術": "技術",
            "未来概念": "科学",
            "専門用語": "その他",
            "新造語": "文化"
        }
        
        mapped_category = category_map.get(category, "その他")
        if mapped_category in self.known_categories:
            similar_concepts = self.known_categories[mapped_category][:5]
            similar_concept = random.choice(similar_concepts) if similar_concepts else "既知の概念"
        else:
            similar_concept = "一般的な概念"
        
        # 推論を生成
        inferences = [
            "同様の原理に基づいて動作する",
            "似た問題を解決するために開発された",
            "共通の理論的基盤を持つ"
        ]
        inference = random.choice(inferences)
        
        template = self.meta_reasoning_patterns[2]["template"]
        return template.format(
            unknown_term=term,
            similar_concept=similar_concept,
            inference=inference
        )
    
    def _generate_knowledge_gap(self, term: str, category: str) -> str:
        """知識ギャップ認識パターンを生成"""
        # 連想を生成
        if "システム" in term:
            association = "何らかの組織化された構造"
        elif "理論" in term:
            association = "学術的な説明体系"
        elif "ネットワーク" in term:
            association = "相互接続された要素の集合"
        else:
            association = "新しい概念的枠組み"
        
        template = self.meta_reasoning_patterns[3]["template"]
        return template.format(
            unknown_term=term,
            association=association
        )
    
    def _generate_hypothesis(self, term: str, category: str) -> str:
        """仮説形成パターンを生成"""
        # カテゴリの推測
        category_guesses = {
            "架空技術": "先進的な技術分野",
            "未来概念": "将来的な発展の方向性",
            "専門用語": "特定分野の専門知識",
            "新造語": "新しい現象や概念"
        }
        category_guess = category_guesses.get(category, "未分類の概念")
        
        # 性質の推測
        property_guesses = [
            "高度な抽象化を伴う",
            "複数の要素を統合する",
            "従来の枠組みを超越する"
        ]
        property_guess = random.choice(property_guesses)
        
        # 検証方法
        verification_methods = [
            "関連文献の調査",
            "専門家への問い合わせ",
            "実験的な検証"
        ]
        verification_method = random.choice(verification_methods)
        
        template = self.meta_reasoning_patterns[4]["template"]
        return template.format(
            unknown_term=term,
            category_guess=category_guess,
            property_guess=property_guess,
            verification_method=verification_method
        )
    
    def _generate_metacognitive(self, term: str, category: str) -> str:
        """メタ認知的反省パターンを生成"""
        # 文脈の推測
        context_guesses = {
            "架空技術": "未来の技術開発",
            "未来概念": "社会の進化",
            "専門用語": "学術研究",
            "新造語": "文化的革新"
        }
        context_guess = context_guesses.get(category, "一般的な議論")
        
        # 既知概念を選択
        if self.known_categories:
            all_concepts = []
            for concepts in self.known_categories.values():
                all_concepts.extend(concepts)
            known_concept = random.choice(all_concepts[:20]) if all_concepts else "基本的な概念"
        else:
            known_concept = "基本的な概念"
        
        template = self.meta_reasoning_patterns[5]["template"]
        return template.format(
            unknown_term=term,
            context_guess=context_guess,
            known_concept=known_concept
        )
    
    def _generate_exploratory(self, term: str, category: str) -> str:
        """探索的思考パターンを生成"""
        # 要素への分解
        if len(term) > 6:
            parts = [term[:len(term)//2], term[len(term)//2:]]
            breakdown = f"「{parts[0]}」と「{parts[1]}」"
        else:
            breakdown = "基本的な構成要素"
        
        # 統合
        syntheses = [
            "新しい統合的概念",
            "革新的なアプローチ",
            "複合的なシステム"
        ]
        synthesis = random.choice(syntheses)
        
        template = self.meta_reasoning_patterns[6]["template"]
        return template.format(
            unknown_term=term,
            breakdown=breakdown,
            synthesis=synthesis
        )
    
    def _generate_uncertainty(self, term: str, category: str) -> str:
        """不確実性表現パターンを生成"""
        # 大胆な推測
        wild_guesses = [
            "革命的な新技術",
            "パラダイムシフトを起こす概念",
            "未知の現象を説明する理論"
        ]
        wild_guess = random.choice(wild_guesses)
        
        # 学習戦略
        learning_strategies = [
            "段階的な情報収集",
            "類似概念からの推論",
            "実践的な探索"
        ]
        learning_strategy = random.choice(learning_strategies)
        
        template = self.meta_reasoning_patterns[7]["template"]
        return template.format(
            unknown_term=term,
            wild_guess=wild_guess,
            learning_strategy=learning_strategy
        )
    
    def generate_with_llm(self, unknown_term: str) -> List[Dict]:
        """ローカルLLMを使用してメタ推論を生成"""
        meta_reasonings = []
        
        prompt = f"""以下の未知の用語について、メタ認知的な推論を3つ生成してください。
未知の用語: {unknown_term}

要件:
- この用語を知らないことを明確に認識する
- 既知の概念から類推して推測する
- 不確実性を認めつつ、学習の意欲を示す
- 「知らない」「学習していない」「推測」「かもしれない」などの表現を使用

形式:
各推論は独立した段落として、異なるアプローチで記述してください。

メタ推論:"""
        
        try:
            response = requests.post(
                self.llm_endpoint,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.9,
                        "top_p": 0.95,
                        "max_tokens": 600
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                # 生成されたテキストを段落に分割
                paragraphs = [p.strip() for p in generated_text.split('\n\n') if p.strip()]
                
                for i, paragraph in enumerate(paragraphs[:3]):
                    meta_reasonings.append({
                        "type": "llm_generated",
                        "content": paragraph,
                        "unknown_term": unknown_term,
                        "category": "llm_generated"
                    })
                    
        except Exception as e:
            print(f"LLM generation error: {e}")
        
        return meta_reasonings
    
    def process_phase4_dataset(self):
        """フェーズ4データセット生成のメインプロセス"""
        output_file = os.path.join(self.output_dir, "meta_reasoning_dataset.jsonl")
        
        # 未知の用語を生成
        unknown_terms = self.generate_unknown_terms(count=500)
        
        all_meta_reasonings = []
        
        with open(output_file, "w", encoding="utf-8") as f:
            for term_data in tqdm(unknown_terms, desc="Generating meta-reasoning"):
                # 1. テンプレートベースのメタ推論生成
                template_reasonings = self.generate_meta_reasoning(term_data)
                
                for reasoning in template_reasonings:
                    reasoning["id"] = f"{term_data['term']}_meta_{reasoning['type']}_{len(all_meta_reasonings)}"
                    f.write(json.dumps(reasoning, ensure_ascii=False) + "\n")
                    all_meta_reasonings.append(reasoning)
                
                # 2. LLMベースのメタ推論生成（オプション）
                if random.random() < 0.1:  # 10%の確率でLLM生成
                    llm_reasonings = self.generate_with_llm(term_data["term"])
                    for i, reasoning in enumerate(llm_reasonings):
                        reasoning["id"] = f"{term_data['term']}_llm_meta_{i}"
                        f.write(json.dumps(reasoning, ensure_ascii=False) + "\n")
                        all_meta_reasonings.append(reasoning)
        
        print(f"\nGenerated {len(all_meta_reasonings)} meta-reasoning processes")
        return all_meta_reasonings
    
    def create_training_sequences(self, tokenizer, max_length: int = 1024):
        """メタ推論データから学習用シーケンスを作成"""
        input_file = os.path.join(self.output_dir, "meta_reasoning_dataset.jsonl")
        output_file = os.path.join(self.output_dir, "training_sequences.pt")
        
        sequences = []
        
        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Creating training sequences"):
                entry = json.loads(line)
                
                # メタ推論テキストをトークナイズ
                text = entry["content"]
                tokens = tokenizer.encode(text, add_special_tokens=True)
                
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
    
    def generate_phase4_dataset(self):
        """フェーズ4データセット生成のメインメソッド"""
        print("\n=== Phase 4: Generating Meta-Reasoning Dataset ===")
        
        if not self.known_concepts:
            print("Warning: No Phase 1 data found. Generating with default known concepts.")
        
        # メタ推論データセットを生成
        meta_reasonings = self.process_phase4_dataset()
        
        # 統計情報を表示
        self.show_statistics()
        
        print("Phase 4 dataset generation completed!")
    
    def show_statistics(self):
        """生成したデータセットの統計情報を表示"""
        input_file = os.path.join(self.output_dir, "meta_reasoning_dataset.jsonl")
        
        stats = {
            "total_reasonings": 0,
            "types": {},
            "unknown_terms": set(),
            "avg_reasoning_length": 0
        }
        
        total_length = 0
        
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                stats["total_reasonings"] += 1
                
                reasoning_type = entry["type"]
                stats["types"][reasoning_type] = stats["types"].get(reasoning_type, 0) + 1
                
                stats["unknown_terms"].add(entry["unknown_term"])
                
                total_length += len(entry["content"])
        
        stats["avg_reasoning_length"] = total_length / max(stats["total_reasonings"], 1)
        stats["unique_unknown_terms"] = len(stats["unknown_terms"])
        
        print("\n=== Dataset Statistics ===")
        print(f"Total meta-reasoning processes: {stats['total_reasonings']}")
        print(f"Unique unknown terms: {stats['unique_unknown_terms']}")
        print(f"Average reasoning length: {stats['avg_reasoning_length']:.0f} characters")
        print("\nReasoning types:")
        for rtype, count in sorted(stats["types"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {rtype}: {count}")


if __name__ == "__main__":
    # テスト実行
    generator = Phase4MetaReasoningDatasetGenerator()
    generator.generate_phase4_dataset()