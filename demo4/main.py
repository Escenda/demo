#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-2 日本語 4フェーズ学習パイプライン
WikipediaやCC-100などから百科事典的な説明文を抽出・生成
"""

import os
import sys
import argparse
import torch
from datasets import load_dataset
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), 'phases'))
from phase1_knowledge import Phase1KnowledgeDatasetGenerator
from phase2_conversation import Phase2ConversationDatasetGenerator
from phase3_thinking import Phase3ThinkingDatasetGenerator
from phase4_meta_reasoning import Phase4MetaReasoningDatasetGenerator

from tokenizer import JapaneseTokenizer
from train import train_phase
from model import get_gpt2_model


def prepare_tokenizer(corpus_path: Optional[str] = None):
    """日本語トークナイザーの準備"""
    print("\n=== Preparing Japanese Tokenizer ===")
    
    tokenizer_path = "tokenizer/japanese_bpe.model"
    tokenizer_dir = os.path.dirname(tokenizer_path)
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    tokenizer = JapaneseTokenizer()
    
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer.load(tokenizer_path)
    else:
        print("Training new tokenizer...")
        
        # Wikipediaコーパスが指定されていればそれを使う
        if corpus_path:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            print("Downloading Japanese Wikipedia for tokenizer training...")
            dataset = load_dataset("graelo/wikipedia", "20230901.ja", split="train[:10000]")
            texts = [item["text"] for item in dataset]
        
        # トークナイザーの学習
        tokenizer.train(texts, model_prefix="tokenizer/japanese_bpe", vocab_size=32000)
        tokenizer.save("tokenizer/config.json")
        
    print(f"Tokenizer ready with vocabulary size: {len(tokenizer)}")
    return tokenizer


def generate_phase1_data(max_samples: Optional[int] = None):
    """フェーズ1: 知識データセット生成"""
    print("\n=== Phase 1: Knowledge Dataset Generation ===")
    
    generator = Phase1KnowledgeDatasetGenerator(
        output_dir="data/phase1",
        max_samples=max_samples
    )
    generator.generate_phase1_dataset()
    
    # トークナイザーで学習用シーケンスを作成
    tokenizer = JapaneseTokenizer("tokenizer/japanese_bpe.model")
    generator.create_training_sequences(tokenizer)


def generate_phase2_data():
    """フェーズ2: 会話データセット生成"""
    print("\n=== Phase 2: Conversation Dataset Generation ===")
    
    generator = Phase2ConversationDatasetGenerator(
        phase1_dir="data/phase1",
        output_dir="data/phase2"
    )
    generator.generate_phase2_dataset()
    
    # トークナイザーで学習用シーケンスを作成
    tokenizer = JapaneseTokenizer("tokenizer/japanese_bpe.model")
    generator.create_training_sequences(tokenizer)


def generate_phase3_data():
    """フェーズ3: 思考過程データセット生成"""
    print("\n=== Phase 3: Thinking Process Dataset Generation ===")
    
    generator = Phase3ThinkingDatasetGenerator(
        phase1_dir="data/phase1",
        output_dir="data/phase3"
    )
    generator.generate_phase3_dataset()
    
    # トークナイザーで学習用シーケンスを作成
    tokenizer = JapaneseTokenizer("tokenizer/japanese_bpe.model")
    generator.create_training_sequences(tokenizer)


def generate_phase4_data():
    """フェーズ4: メタ推論データセット生成"""
    print("\n=== Phase 4: Meta-Reasoning Dataset Generation ===")
    
    generator = Phase4MetaReasoningDatasetGenerator(
        phase1_dir="data/phase1",
        output_dir="data/phase4"
    )
    generator.generate_phase4_dataset()
    
    # トークナイザーで学習用シーケンスを作成
    tokenizer = JapaneseTokenizer("tokenizer/japanese_bpe.model")
    generator.create_training_sequences(tokenizer)


def run_training():
    """4フェーズの学習を実行"""
    print("\n=== Starting 4-Phase Training ===")
    
    # train.pyのmain関数を呼び出し
    from train import main as train_main
    
    # main.pyで定義した設定情報を渡す
    phase_configs = get_phase_configs()
    train_main(phase_configs)


def generate_text(checkpoint_path: str, prompt: str, max_length: int = 100):
    """テキスト生成を実行"""
    print("\n=== Text Generation ===")
    
    # デバイスの選択
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # トークナイザーとモデルの準備
    tokenizer = JapaneseTokenizer("tokenizer/japanese_bpe.model")
    model = get_gpt2_model(model_size="medium", vocab_size=len(tokenizer))
    
    # チェックポイントからモデルをロード
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model.pt")))
    model = model.to(device)
    model.eval()
    
    # 入力文をトークナイズ
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)]).to(device)
    
    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.8,
            top_k=50,
            top_p=0.95
        )
    
    # デコード
    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    return generated_text


def get_phase_configs():
    """全フェーズの学習設定を取得"""
    configs = {
        "phase1": {
            "description": "Knowledge Explanation Learning",
            "train_data_path": "data/phase1/training_sequences.pt",
            "output_dir": "checkpoints/phase1",
            "tokenizer_path": "tokenizer/japanese_bpe.model",
            "model_size": "medium",
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "warmup_steps": 2000
        },
        "phase2": {
            "description": "Conversation Learning",
            "train_data_path": "data/phase2/training_sequences.pt",
            "output_dir": "checkpoints/phase2",
            "tokenizer_path": "tokenizer/japanese_bpe.model",
            "previous_checkpoint": "checkpoints/phase1/checkpoint_best",
            "learning_rate": 1e-4,
            "num_epochs": 5,
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "warmup_steps": 1000
        },
        "phase3": {
            "description": "Thinking Process Learning",
            "train_data_path": "data/phase3/training_sequences.pt",
            "output_dir": "checkpoints/phase3",
            "tokenizer_path": "tokenizer/japanese_bpe.model",
            "previous_checkpoint": "checkpoints/phase2/checkpoint_best",
            "learning_rate": 5e-5,
            "num_epochs": 8,
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "warmup_steps": 500
        },
        "phase4": {
            "description": "Meta-Reasoning Learning",
            "train_data_path": "data/phase4/training_sequences.pt",
            "output_dir": "checkpoints/phase4",
            "tokenizer_path": "tokenizer/japanese_bpe.model",
            "previous_checkpoint": "checkpoints/phase3/checkpoint_best",
            "learning_rate": 2e-5,
            "num_epochs": 10,
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "warmup_steps": 200
        }
    }
    return configs


def main():
    """コマンドライン引数の処理"""
    parser = argparse.ArgumentParser(description="GPT-2 Japanese 4-Phase Training Pipeline")
    
    # サブコマンドの設定
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # データ生成コマンド
    data_parser = subparsers.add_parser('generate-data', help='Generate datasets for all phases')
    data_parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4], 
                           help='Generate data for specific phase only')
    data_parser.add_argument('--max-samples', type=int, 
                           help='Maximum samples for phase 1 (for testing)')
    
    # トークナイザー準備コマンド
    tokenizer_parser = subparsers.add_parser('prepare-tokenizer', help='Prepare Japanese tokenizer')
    tokenizer_parser.add_argument('--corpus-path', type=str, 
                                help='Path to corpus file for tokenizer training')
    
    # 学習コマンド
    train_parser = subparsers.add_parser('train', help='Run 4-phase training')
    train_parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4],
                            help='Train specific phase only')
    
    # 生成コマンド
    generate_parser = subparsers.add_parser('generate', help='Generate text with trained model')
    generate_parser.add_argument('--checkpoint', type=str, required=True,
                               help='Path to model checkpoint')
    generate_parser.add_argument('--prompt', type=str, required=True,
                               help='Input prompt for generation')
    generate_parser.add_argument('--max-length', type=int, default=100,
                               help='Maximum generation length')
    
    # フルパイプライン実行コマンド
    full_parser = subparsers.add_parser('run-all', help='Run complete pipeline')
    full_parser.add_argument('--max-samples', type=int,
                           help='Maximum samples for phase 1 (for testing)')
    
    args = parser.parse_args()
    
    if args.command == 'prepare-tokenizer':
        prepare_tokenizer(args.corpus_path)
        
    elif args.command == 'generate-data':
        # トークナイザーがなければ作成
        if not os.path.exists("tokenizer/japanese_bpe.model"):
            prepare_tokenizer()
        
        if args.phase:
            # 指定フェーズのみ生成
            if args.phase == 1:
                generate_phase1_data(args.max_samples)
            elif args.phase == 2:
                generate_phase2_data()
            elif args.phase == 3:
                generate_phase3_data()
            elif args.phase == 4:
                generate_phase4_data()
        else:
            # 全フェーズ生成
            generate_phase1_data(args.max_samples)
            generate_phase2_data()
            generate_phase3_data()
            generate_phase4_data()
    
    elif args.command == 'train':
        if args.phase:
            # 指定フェーズのみ学習
            phase_configs = get_phase_configs()
            if f"phase{args.phase}" in phase_configs:
                config = phase_configs[f"phase{args.phase}"]
                train_phase(args.phase, config)
            else:
                print(f"Error: Configuration for phase {args.phase} not found.")
        else:
            # 全フェーズ学習
            run_training()
    
    elif args.command == 'generate':
        generate_text(args.checkpoint, args.prompt, args.max_length)
    
    elif args.command == 'run-all':
        # 全体パイプライン実行
        print("Running complete 4-phase training pipeline...")
        
        # 1. トークナイザー準備
        prepare_tokenizer()
        
        # 2. データ生成
        generate_phase1_data(args.max_samples)
        generate_phase2_data()
        generate_phase3_data()
        generate_phase4_data()
        
        # 3. 学習
        run_training()
        
        print("\nPipeline completed successfully!")
        print("Trained model checkpoints are available in 'checkpoints/' directory")
        print("To generate text, run:")
        print("  python main.py generate --checkpoint checkpoints/phase4/checkpoint_best --prompt 'your prompt here'")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()