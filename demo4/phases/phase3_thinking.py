import os
import json
import random
import re
from typing import List, Dict, Optional
from tqdm import tqdm
import requests


class Phase3ThinkingDatasetGenerator:
    """
    フェーズ3: 既知情報に基づく手続き的・内省的思考過程データセット生成
    チェイン・オブ・ソート（Chain-of-Thought）形式の推論文を生成
    """
    
    def __init__(self, phase1_dir: str = "data/phase1", output_dir: str = "data/phase3"):
        self.phase1_dir = phase1_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # フェーズ1の知識を読み込む
        self.phase1_knowledge = []
        self._load_phase1_data()
        
        # ローカルLLM設定
        self.llm_endpoint = "http://localhost:11434/api/generate"
        self.llm_model = "gemma3:4b-it-qat"
        
        # 思考パターンテンプレート
        self.thinking_patterns = self._init_thinking_patterns()
    
    def _load_phase1_data(self):
        """フェーズ1のデータを読み込む"""
        phase1_file = os.path.join(self.phase1_dir, "knowledge_dataset.jsonl")
        
        if not os.path.exists(phase1_file):
            print(f"Warning: Phase 1 data not found at {phase1_file}")
            return
        
        with open(phase1_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                self.phase1_knowledge.append(entry)
        
        print(f"Loaded {len(self.phase1_knowledge)} knowledge entries for thinking process")
    
    def _init_thinking_patterns(self) -> List[Dict]:
        """思考パターンテンプレートを初期化"""
        return [
            {
                "type": "why_reasoning",
                "template": "私は今、{topic}について考えている。{fact}ということを知っている。なぜ{topic}がそのような性質を持つのか考えてみよう。{reasoning}。だから、{conclusion}ということが分かった。"
            },
            {
                "type": "how_procedure",
                "template": "{goal}するにはどうすればよいか考えてみる。まず、{step1}する必要がある。次に、{step2}を行う。最後に、{step3}することで、{goal}が達成できるだろう。"
            },
            {
                "type": "comparison_analysis",
                "template": "{item1}と{item2}を比較して考えてみよう。{item1}は{feature1}という特徴がある。一方、{item2}は{feature2}という点で異なる。これらを総合すると、{insight}ということが言える。"
            },
            {
                "type": "causal_chain",
                "template": "{cause}について深く考察してみる。{cause}が起こると、{effect1}という結果が生じる。さらに、{effect1}は{effect2}を引き起こす。このような因果の連鎖から、{final_insight}ということが理解できる。"
            },
            {
                "type": "problem_solving",
                "template": "{problem}という問題について考えている。この問題の本質は{essence}にある。解決策として、{solution1}という方法が考えられる。また、{solution2}というアプローチも有効だろう。最適な解決策は{best_solution}だと思われる。"
            },
            {
                "type": "abstraction",
                "template": "{concrete}という具体的な事例から、より抽象的な原理を導き出してみよう。{concrete}の本質は{essence}にある。これを一般化すると、{abstract_principle}という原理が見えてくる。この原理は{application}にも応用できるはずだ。"
            },
            {
                "type": "hypothesis_testing",
                "template": "{hypothesis}という仮説を立ててみた。この仮説が正しいとすると、{prediction1}が予測される。また、{prediction2}も起こるはずだ。既知の事実{fact}と照らし合わせると、{evaluation}。よって、この仮説は{conclusion}と考えられる。"
            },
            {
                "type": "systematic_analysis",
                "template": "{subject}を体系的に分析してみよう。まず、構成要素として{component1}、{component2}、{component3}がある。これらの関係性を見ると、{relationship}という構造が見えてくる。全体として、{holistic_view}という理解に至った。"
            }
        ]
    
    def generate_thinking_process(self, knowledge_entry: Dict) -> List[Dict]:
        """知識エントリから思考過程を生成"""
        thinking_processes = []
        title = knowledge_entry["title"]
        explanation = knowledge_entry["full_explanation"]
        category = knowledge_entry.get("category", "その他")
        
        # 各思考パターンを適用
        selected_patterns = random.sample(self.thinking_patterns, min(4, len(self.thinking_patterns)))
        
        for pattern in selected_patterns:
            try:
                if pattern["type"] == "why_reasoning":
                    process = self._generate_why_reasoning(title, explanation)
                elif pattern["type"] == "how_procedure":
                    process = self._generate_how_procedure(title, explanation, category)
                elif pattern["type"] == "comparison_analysis":
                    process = self._generate_comparison_analysis(knowledge_entry)
                elif pattern["type"] == "causal_chain":
                    process = self._generate_causal_chain(title, explanation)
                elif pattern["type"] == "problem_solving":
                    process = self._generate_problem_solving(title, category)
                elif pattern["type"] == "abstraction":
                    process = self._generate_abstraction(title, explanation)
                elif pattern["type"] == "hypothesis_testing":
                    process = self._generate_hypothesis(title, explanation)
                elif pattern["type"] == "systematic_analysis":
                    process = self._generate_systematic_analysis(title, explanation)
                else:
                    continue
                
                if process:
                    thinking_processes.append({
                        "type": pattern["type"],
                        "content": process,
                        "source_knowledge": title
                    })
                    
            except Exception as e:
                continue
        
        return thinking_processes
    
    def _generate_why_reasoning(self, topic: str, explanation: str) -> str:
        """Why型の推論を生成"""
        # 説明文から事実を抽出
        fact = self._extract_fact(explanation)
        
        # 推論プロセスを構築
        reasoning_steps = [
            f"{topic}の性質を考えると",
            "この特徴は特定の条件下で生じると考えられる",
            "さらに深く分析すると、根本的な原因が見えてくる"
        ]
        reasoning = "。".join(reasoning_steps)
        
        # 結論を生成
        conclusion = f"{topic}の本質的な性質が明らかになった"
        
        return f"私は今、{topic}について考えている。{fact}ということを知っている。なぜ{topic}がそのような性質を持つのか考えてみよう。{reasoning}。だから、{conclusion}ということが分かった。"
    
    def _generate_how_procedure(self, topic: str, explanation: str, category: str) -> str:
        """How型の手順を生成"""
        # カテゴリに応じた目標を設定
        goals = {
            "科学": f"{topic}を理解",
            "技術": f"{topic}を実装",
            "歴史": f"{topic}について調査",
            "文化": f"{topic}を体験",
            "社会": f"{topic}に参加",
            "地理": f"{topic}を訪問",
            "人物": f"{topic}について学習"
        }
        goal = goals.get(category, f"{topic}を理解")
        
        # 手順を生成
        steps = [
            f"基本的な概念を把握",
            f"詳細な情報を収集",
            f"実践的な応用を検討"
        ]
        
        return f"{goal}するにはどうすればよいか考えてみる。まず、{steps[0]}する必要がある。次に、{steps[1]}を行う。最後に、{steps[2]}することで、{goal}が達成できるだろう。"
    
    def _generate_comparison_analysis(self, knowledge_entry: Dict) -> Optional[str]:
        """比較分析を生成"""
        if len(self.phase1_knowledge) < 2:
            return None
        
        # 同じカテゴリから別のアイテムを選択
        same_category = [k for k in self.phase1_knowledge 
                        if k.get("category") == knowledge_entry.get("category") 
                        and k["title"] != knowledge_entry["title"]]
        
        if not same_category:
            other_entry = random.choice(self.phase1_knowledge)
        else:
            other_entry = random.choice(same_category)
        
        item1 = knowledge_entry["title"]
        item2 = other_entry["title"]
        
        # 特徴を抽出
        feature1 = self._extract_feature(knowledge_entry["full_explanation"])
        feature2 = self._extract_feature(other_entry["full_explanation"])
        
        # 洞察を生成
        insight = "それぞれ独自の特徴を持ちながら、共通の側面も存在する"
        
        return f"{item1}と{item2}を比較して考えてみよう。{item1}は{feature1}という特徴がある。一方、{item2}は{feature2}という点で異なる。これらを総合すると、{insight}ということが言える。"
    
    def _generate_causal_chain(self, topic: str, explanation: str) -> str:
        """因果連鎖の推論を生成"""
        # 原因を設定
        cause = f"{topic}の基本的な性質"
        
        # 効果の連鎖を生成
        effects = [
            "直接的な影響が現れる",
            "さらに二次的な変化が生じる",
            "最終的に全体的な変化につながる"
        ]
        
        final_insight = f"{topic}の影響は多層的で複雑である"
        
        return f"{cause}について深く考察してみる。{cause}が起こると、{effects[0]}という結果が生じる。さらに、{effects[0]}は{effects[1]}を引き起こす。このような因果の連鎖から、{final_insight}ということが理解できる。"
    
    def _generate_problem_solving(self, topic: str, category: str) -> str:
        """問題解決型の思考を生成"""
        # カテゴリに応じた問題を設定
        problems = {
            "科学": f"{topic}に関する未解明の現象",
            "技術": f"{topic}の実装上の課題",
            "社会": f"{topic}に関する社会的課題",
            "文化": f"{topic}の継承問題",
            "その他": f"{topic}に関する一般的な課題"
        }
        
        problem = problems.get(category, problems["その他"])
        essence = "根本的な理解の不足"
        
        solutions = [
            "体系的なアプローチ",
            "段階的な改善",
            "総合的な取り組み"
        ]
        
        return f"{problem}という問題について考えている。この問題の本質は{essence}にある。解決策として、{solutions[0]}という方法が考えられる。また、{solutions[1]}というアプローチも有効だろう。最適な解決策は{solutions[2]}だと思われる。"
    
    def _generate_abstraction(self, topic: str, explanation: str) -> str:
        """抽象化の思考を生成"""
        concrete = f"{topic}の具体例"
        essence = "基本的な原理"
        abstract_principle = "より一般的な法則"
        application = "他の分野"
        
        return f"{concrete}という具体的な事例から、より抽象的な原理を導き出してみよう。{concrete}の本質は{essence}にある。これを一般化すると、{abstract_principle}という原理が見えてくる。この原理は{application}にも応用できるはずだ。"
    
    def _generate_hypothesis(self, topic: str, explanation: str) -> str:
        """仮説検証型の思考を生成"""
        hypothesis = f"{topic}には未知の側面がある"
        predictions = [
            "新たな発見が期待できる",
            "既存の理解が深まる"
        ]
        
        fact = self._extract_fact(explanation)
        evaluation = "仮説と事実が整合する"
        conclusion = "有望である"
        
        return f"{hypothesis}という仮説を立ててみた。この仮説が正しいとすると、{predictions[0]}が予測される。また、{predictions[1]}も起こるはずだ。既知の事実{fact}と照らし合わせると、{evaluation}。よって、この仮説は{conclusion}と考えられる。"
    
    def _generate_systematic_analysis(self, topic: str, explanation: str) -> str:
        """体系的分析の思考を生成"""
        components = [
            "基本要素",
            "関連要素",
            "周辺要素"
        ]
        
        relationship = "相互に影響し合う関係"
        holistic_view = f"{topic}は複雑なシステムである"
        
        return f"{topic}を体系的に分析してみよう。まず、構成要素として{components[0]}、{components[1]}、{components[2]}がある。これらの関係性を見ると、{relationship}という構造が見えてくる。全体として、{holistic_view}という理解に至った。"
    
    def _extract_fact(self, text: str) -> str:
        """テキストから事実を抽出"""
        sentences = re.split(r'[。．]', text)
        for sent in sentences[:5]:
            if len(sent) > 20 and len(sent) < 100:
                return sent.strip() + "。"
        return "基本的な事実がある。"
    
    def _extract_feature(self, text: str) -> str:
        """テキストから特徴を抽出"""
        # 「〜性」「〜的」などの特徴語を探す
        patterns = [
            r'(\w+性)',
            r'(\w+的)',
            r'(\w+という特徴)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        
        return "独自の特徴"
    
    def generate_with_llm(self, knowledge_entry: Dict) -> List[Dict]:
        """ローカルLLMを使用して思考過程を生成"""
        thinking_processes = []
        
        prompt = f"""以下の知識に基づいて、内省的な思考過程を3つ生成してください。
知識: {knowledge_entry['full_explanation'][:500]}

要件:
- 既存の知識のみを使用
- 段階的な推論過程を含む
- 「なぜ」「どのように」「もし〜なら」などの思考パターンを使用
- 1人称の内省的な文体

形式:
各思考過程は独立した段落として記述してください。

思考過程:"""
        
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
                        "max_tokens": 800
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                # 生成されたテキストを段落に分割
                paragraphs = [p.strip() for p in generated_text.split('\n\n') if p.strip()]
                
                for i, paragraph in enumerate(paragraphs[:3]):
                    thinking_processes.append({
                        "type": "llm_generated",
                        "content": paragraph,
                        "source_knowledge": knowledge_entry["title"]
                    })
                    
        except Exception as e:
            print(f"LLM generation error: {e}")
        
        return thinking_processes
    
    def process_phase3_dataset(self):
        """フェーズ3データセット生成のメインプロセス"""
        output_file = os.path.join(self.output_dir, "thinking_dataset.jsonl")
        
        all_thinking_processes = []
        
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in tqdm(self.phase1_knowledge, desc="Generating thinking processes"):
                # 1. テンプレートベースの思考過程生成
                template_processes = self.generate_thinking_process(entry)
                
                for process in template_processes:
                    process["id"] = f"{entry['title']}_think_{process['type']}_{len(all_thinking_processes)}"
                    f.write(json.dumps(process, ensure_ascii=False) + "\n")
                    all_thinking_processes.append(process)
                
                # 2. LLMベースの思考過程生成（オプション）
                if random.random() < 0.2:  # 20%の確率でLLM生成
                    llm_processes = self.generate_with_llm(entry)
                    for i, process in enumerate(llm_processes):
                        process["id"] = f"{entry['title']}_llm_think_{i}"
                        f.write(json.dumps(process, ensure_ascii=False) + "\n")
                        all_thinking_processes.append(process)
        
        print(f"\nGenerated {len(all_thinking_processes)} thinking processes")
        return all_thinking_processes
    
    def create_training_sequences(self, tokenizer, max_length: int = 1024):
        """思考過程データから学習用シーケンスを作成"""
        input_file = os.path.join(self.output_dir, "thinking_dataset.jsonl")
        output_file = os.path.join(self.output_dir, "training_sequences.pt")
        
        sequences = []
        
        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Creating training sequences"):
                entry = json.loads(line)
                
                # 思考過程テキストをトークナイズ
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
    
    def generate_phase3_dataset(self):
        """フェーズ3データセット生成のメインメソッド"""
        print("\n=== Phase 3: Generating Thinking Process Dataset ===")
        
        if not self.phase1_knowledge:
            print("Error: No Phase 1 data found. Please run Phase 1 first.")
            return
        
        # 思考過程データセットを生成
        thinking_processes = self.process_phase3_dataset()
        
        # 統計情報を表示
        self.show_statistics()
        
        print("Phase 3 dataset generation completed!")
    
    def show_statistics(self):
        """生成したデータセットの統計情報を表示"""
        input_file = os.path.join(self.output_dir, "thinking_dataset.jsonl")
        
        stats = {
            "total_processes": 0,
            "types": {},
            "avg_process_length": 0
        }
        
        total_length = 0
        
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                stats["total_processes"] += 1
                
                process_type = entry["type"]
                stats["types"][process_type] = stats["types"].get(process_type, 0) + 1
                
                total_length += len(entry["content"])
        
        stats["avg_process_length"] = total_length / max(stats["total_processes"], 1)
        
        print("\n=== Dataset Statistics ===")
        print(f"Total thinking processes: {stats['total_processes']}")
        print(f"Average process length: {stats['avg_process_length']:.0f} characters")
        print("\nProcess types:")
        for ptype, count in sorted(stats["types"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {ptype}: {count}")


if __name__ == "__main__":
    # テスト実行
    generator = Phase3ThinkingDatasetGenerator()
    generator.generate_phase3_dataset()