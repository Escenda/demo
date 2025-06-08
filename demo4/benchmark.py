#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマークとプロファイリングツール
"""

import os
import time
import torch
import psutil
import gc
from datetime import datetime
import json
import matplotlib.pyplot as plt
from contextlib import contextmanager
from model import get_gpt2_model
from tokenizer import JapaneseTokenizer
from train import GPT2Dataset, Trainer
from train_optimized import OptimizedTrainer


class BenchmarkProfiler:
    """学習のベンチマークとプロファイリング"""
    
    def __init__(self, output_dir="benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
    
    @contextmanager
    def profile_section(self, name):
        """セクションのプロファイリング"""
        start_time = time.time()
        start_memory = 0
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        
        yield
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            memory_used = (end_memory - start_memory) / 1024**3
        else:
            memory_used = 0
        
        elapsed_time = time.time() - start_time
        
        self.results.append({
            'name': name,
            'time': elapsed_time,
            'memory_gb': memory_used,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"[{name}] Time: {elapsed_time:.2f}s, Memory: {memory_used:.2f}GB")
    
    def benchmark_model_loading(self, model_size="small", vocab_size=32000):
        """モデル読み込みのベンチマーク"""
        print("\n=== Benchmarking Model Loading ===")
        
        # 通常のモデル読み込み
        with self.profile_section("Normal Model Loading"):
            model = get_gpt2_model(model_size=model_size, vocab_size=vocab_size)
            if torch.cuda.is_available():
                model = model.cuda()
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        # グラディエントチェックポイント付きモデル読み込み
        with self.profile_section("Model with Gradient Checkpointing"):
            model = get_gpt2_model(
                model_size=model_size, 
                vocab_size=vocab_size,
                use_gradient_checkpointing=True
            )
            if torch.cuda.is_available():
                model = model.cuda()
        
        return model
    
    def benchmark_forward_pass(self, model, batch_size=4, seq_length=512):
        """順伝播のベンチマーク"""
        print("\n=== Benchmarking Forward Pass ===")
        
        device = next(model.parameters()).device
        input_ids = torch.randint(0, 32000, (batch_size, seq_length)).to(device)
        
        # 通常の順伝播
        model.eval()
        with self.profile_section("Normal Forward Pass"):
            with torch.no_grad():
                outputs = model(input_ids)
        
        # Mixed Precision順伝播
        with self.profile_section("Mixed Precision Forward Pass"):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids)
        
        # 学習モードでの順伝播（グラディエント計算あり）
        model.train()
        with self.profile_section("Training Forward Pass"):
            outputs = model(input_ids, labels=input_ids)
            loss = outputs["loss"]
        
        return loss
    
    def benchmark_backward_pass(self, model, loss):
        """逆伝播のベンチマーク"""
        print("\n=== Benchmarking Backward Pass ===")
        
        # 通常の逆伝播
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        with self.profile_section("Normal Backward Pass"):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # 最適化された逆伝播
        optimizer_fused = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-4, 
            fused=True if torch.cuda.is_available() else False
        )
        
        # 新しい順伝播を実行
        device = next(model.parameters()).device
        input_ids = torch.randint(0, 32000, (4, 512)).to(device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs["loss"]
        
        with self.profile_section("Optimized Backward Pass"):
            loss.backward()
            optimizer_fused.step()
            optimizer_fused.zero_grad(set_to_none=True)
    
    def benchmark_data_loading(self, data_path, batch_size=4):
        """データローディングのベンチマーク"""
        print("\n=== Benchmarking Data Loading ===")
        
        if not os.path.exists(data_path):
            print(f"Data file not found: {data_path}")
            return
        
        # 通常のデータセット
        with self.profile_section("Normal Dataset Loading"):
            dataset = GPT2Dataset(data_path, use_memory_map=False)
        
        # メモリマップ付きデータセット
        with self.profile_section("Memory-mapped Dataset Loading"):
            dataset_mmap = GPT2Dataset(data_path, use_memory_map=True)
        
        # DataLoaderのベンチマーク
        from torch.utils.data import DataLoader
        
        with self.profile_section("DataLoader (4 workers)"):
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
            for i, batch in enumerate(loader):
                if i >= 10:  # 最初の10バッチのみ
                    break
        
        with self.profile_section("DataLoader (8 workers, pinned memory)"):
            loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                num_workers=8,
                pin_memory=True,
                persistent_workers=True
            )
            for i, batch in enumerate(loader):
                if i >= 10:
                    break
    
    def compare_training_methods(self, config):
        """通常の学習と最適化された学習の比較"""
        print("\n=== Comparing Training Methods ===")
        
        # 小さなデータセットを作成（テスト用）
        test_data = torch.randint(0, 32000, (100, 512))
        torch.save(test_data, "benchmark_test_data.pt")
        
        # トークナイザーの準備
        tokenizer = JapaneseTokenizer()
        
        # モデルの準備
        model_normal = get_gpt2_model(model_size="small", vocab_size=32000)
        model_optimized = get_gpt2_model(
            model_size="small", 
            vocab_size=32000,
            use_gradient_checkpointing=True
        )
        
        # 通常のトレーナー
        with self.profile_section("Normal Trainer - 10 steps"):
            trainer = Trainer(
                model=model_normal,
                tokenizer=tokenizer,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                output_dir="benchmark_normal",
                fp16=False
            )
            dataset = GPT2Dataset("benchmark_test_data.pt")
            
            # 10ステップだけ学習
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=2, shuffle=True)
            
            for i, batch in enumerate(loader):
                if i >= 10:
                    break
                trainer.train_epoch(loader, epoch=0)
                break
        
        # 最適化されたトレーナー
        with self.profile_section("Optimized Trainer - 10 steps"):
            trainer_opt = OptimizedTrainer(
                model=model_optimized,
                tokenizer=tokenizer,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                output_dir="benchmark_optimized",
                fp16=True,
                use_gradient_checkpointing=True,
                use_fused_optimizer=True
            )
            
            for i, batch in enumerate(loader):
                if i >= 10:
                    break
                trainer_opt.train_epoch(loader, epoch=0)
                break
        
        # クリーンアップ
        os.remove("benchmark_test_data.pt")
    
    def save_results(self):
        """結果を保存"""
        output_file = os.path.join(self.output_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        
        # 結果のサマリーを表示
        print("\n=== Benchmark Summary ===")
        for result in self.results:
            print(f"{result['name']:40s} - Time: {result['time']:6.2f}s, Memory: {result['memory_gb']:6.2f}GB")
    
    def plot_results(self):
        """結果をプロット"""
        if not self.results:
            return
        
        names = [r['name'] for r in self.results]
        times = [r['time'] for r in self.results]
        memories = [r['memory_gb'] for r in self.results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 実行時間のプロット
        ax1.barh(names, times)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.grid(True, alpha=0.3)
        
        # メモリ使用量のプロット
        ax2.barh(names, memories)
        ax2.set_xlabel('Memory (GB)')
        ax2.set_title('Memory Usage Comparison')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {output_file}")


def main():
    """ベンチマークの実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark LLM training optimizations")
    parser.add_argument('--model-size', choices=['small', 'medium', 'large'], default='small',
                       help='Model size to benchmark')
    parser.add_argument('--data-path', type=str, 
                       default='data/phase1/training_sequences.pt',
                       help='Path to training data')
    parser.add_argument('--all', action='store_true',
                       help='Run all benchmarks')
    
    args = parser.parse_args()
    
    profiler = BenchmarkProfiler()
    
    if args.all:
        # すべてのベンチマークを実行
        model = profiler.benchmark_model_loading(model_size=args.model_size)
        loss = profiler.benchmark_forward_pass(model)
        profiler.benchmark_backward_pass(model, loss)
        profiler.benchmark_data_loading(args.data_path)
        profiler.compare_training_methods({})
    else:
        # 個別のベンチマーク
        print("Run with --all to execute all benchmarks")
        print("Available benchmarks:")
        print("  - Model loading")
        print("  - Forward pass")
        print("  - Backward pass") 
        print("  - Data loading")
        print("  - Training method comparison")
    
    profiler.save_results()
    profiler.plot_results()


if __name__ == "__main__":
    main()