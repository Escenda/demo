#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最適化された訓練スクリプト - LLM高速化手法を全て実装
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
import psutil
import gc
from contextlib import contextmanager
from model import get_gpt2_model
from tokenizer import JapaneseTokenizer
from train import GPT2Dataset, Trainer


class OptimizedTrainer(Trainer):
    """最適化されたGPT-2トレーナー"""
    
    def __init__(
        self,
        model,
        tokenizer,
        device='cuda',
        output_dir='checkpoints',
        learning_rate=2e-4,
        warmup_steps=2000,
        gradient_accumulation_steps=16,
        fp16=True,
        max_grad_norm=1.0,
        use_gradient_checkpointing=True,
        use_cpu_offload=False,
        use_fused_optimizer=True,
        compile_model=False
    ):
        super().__init__(
            model, tokenizer, device, output_dir,
            learning_rate, warmup_steps, gradient_accumulation_steps,
            fp16, max_grad_norm
        )
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_cpu_offload = use_cpu_offload
        self.use_fused_optimizer = use_fused_optimizer
        
        # モデルコンパイル（PyTorch 2.0+）
        if compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)
        
        # Fused Optimizer（高速化）
        if use_fused_optimizer and hasattr(optim, 'AdamW'):
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
                fused=True if torch.cuda.is_available() else False
            )
        
        # CPU Offload（大規模モデル用）
        if use_cpu_offload:
            self._setup_cpu_offload()
    
    def _setup_cpu_offload(self):
        """CPU Offloadの設定"""
        # パラメータとグラデイエントのCPUオフロード
        for param in self.model.parameters():
            param.data = param.data.cpu()
            if param.grad is not None:
                param.grad = param.grad.cpu()
    
    @contextmanager
    def _memory_efficient_forward(self):
        """メモリ効率的な順伝播のコンテキストマネージャー"""
        # 不要なキャッシュをクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        yield
        
        # 順伝播後にメモリを解放
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def train_epoch(self, train_loader, epoch):
        """最適化された1エポックの学習"""
        self.model.train()
        total_loss = 0
        step_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        
        # データプリフェッチャー
        data_iter = iter(train_loader)
        
        for i in range(len(train_loader)):
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            
            # バッチデータをデバイスに非同期転送
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # メモリ効率的な順伝播
            with self._memory_efficient_forward():
                # Mixed Precisionを使用
                if self.scaler:
                    with autocast():
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        loss = outputs['loss']
                else:
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs['loss']
                
                # 勾配の蓄積
                loss = loss / self.gradient_accumulation_steps
                
                # バックプロパゲーション
                if self.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            step_loss += loss.item()
            
            # 勾配蓄積が完了したら更新
            if (i + 1) % self.gradient_accumulation_steps == 0:
                # 勾配クリッピング
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # 勾配を効率的にリセット
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                
                # メモリ使用量を記録
                if self.global_step % 100 == 0:
                    self._log_memory_usage()
                
                # ロスを記録
                total_loss += step_loss
                progress_bar.set_postfix({
                    'loss': f'{step_loss:.4f}',
                    'lr': f'{self.get_lr():.2e}',
                    'step': self.global_step,
                    'mem_gb': f'{torch.cuda.memory_allocated() / 1024**3:.1f}' if torch.cuda.is_available() else 'N/A'
                })
                step_loss = 0
                
                # 定期的にチェックポイントを保存
                if self.global_step % 5000 == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
        
        return total_loss / len(train_loader)
    
    def train_with_optimization(
        self,
        train_dataset,
        eval_dataset=None,
        batch_size=2,
        num_epochs=1,
        use_dynamic_batching=True,
        max_tokens_per_batch=None
    ):
        """最適化された学習メソッド"""
        
        # 動的バッチサイズ調整
        if use_dynamic_batching and max_tokens_per_batch:
            from torch.utils.data import Sampler
            # TODO: 実装する動的バッチサンプラー
            pass
        
        # 最適化されたDataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count()),  # CPU数に応じて調整
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True  # 最後の不完全なバッチを削除してパフォーマンスを安定化
        )
        
        # One Cycle LRスケジューラ（高速収束）
        total_steps = len(train_loader) * num_epochs // self.gradient_accumulation_steps
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,  # ウォームアップの割合
            anneal_strategy='cos'
        )
        
        # 学習開始
        print(f"Starting optimized training with {len(train_dataset)} samples")
        print(f"Batch size: {batch_size}, Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * self.gradient_accumulation_steps}")
        print(f"Gradient checkpointing: {self.use_gradient_checkpointing}")
        print(f"Mixed precision: {self.fp16}")
        
        best_eval_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # 学習
            train_loss = self.train_epoch(train_loader, epoch)
            
            # スケジューラを更新
            if hasattr(self, 'scheduler'):
                scheduler.step()
            
            # 評価
            if eval_dataset:
                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=batch_size * 2,  # 評価時は大きめのバッチサイズ
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
                eval_metrics = self.evaluate(eval_loader)
                print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}, "
                      f"Eval Loss: {eval_metrics['loss']:.4f}, "
                      f"Perplexity: {eval_metrics['perplexity']:.2f}")
                
                # ベストモデルを保存
                if eval_metrics['loss'] < best_eval_loss:
                    best_eval_loss = eval_metrics['loss']
                    self.save_checkpoint("best")
            else:
                print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
            
            # メモリサマリーを表示
            mem_summary = self.get_memory_summary()
            if mem_summary:
                print(f"Memory - Allocated: {mem_summary['current_allocated_gb']:.2f}GB, "
                      f"Reserved: {mem_summary['current_reserved_gb']:.2f}GB")
            
            # エポックごとにチェックポイントを保存
            self.save_checkpoint(f"epoch_{epoch}")
            
            elapsed_time = time.time() - start_time
            print(f"Epoch completed in {elapsed_time:.2f} seconds")
            
            # メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def train_phase_optimized(phase, config):
    """最適化された特定フェーズの学習"""
    print(f"\n{'='*50}")
    print(f"Training Phase {phase} (Optimized): {config['description']}")
    print(f"{'='*50}\n")
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # GPUメモリ情報を表示
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    
    # トークナイザーの読み込み
    tokenizer = JapaneseTokenizer()
    if os.path.exists(config.get("tokenizer_path", "tokenizer/japanese_bpe.model")):
        tokenizer.load(config["tokenizer_path"])
    else:
        print("Warning: Tokenizer not found. Please train tokenizer first.")
        return
    
    # モデルの作成（最適化オプション付き）
    model = get_gpt2_model(
        model_size=config.get("model_size", "medium"),
        vocab_size=len(tokenizer),
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", True)
    )
    
    # 前のフェーズのチェックポイントを読み込む
    if config.get("previous_checkpoint"):
        print(f"Loading model from previous checkpoint: {config['previous_checkpoint']}")
        model.load_state_dict(torch.load(os.path.join(config["previous_checkpoint"], "model.pt")))
    
    model = model.to(device)
    
    # モデルパラメータ数を表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # データセットの準備（最適化版）
    try:
        dataset = GPT2Dataset(
            data_path=config["train_data_path"],
            use_memory_map=True  # メモリマップを使用
        )
    except FileNotFoundError:
        print(f"Error: Training data not found at {config['train_data_path']}")
        return
    
    # 最適化されたトレーナーの作成
    trainer = OptimizedTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        fp16=config.get("fp16", True),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", True),
        use_fused_optimizer=config.get("use_fused_optimizer", True),
        compile_model=config.get("compile_model", False)  # PyTorch 2.0+ only
    )
    
    # 評価データセットの準備
    eval_dataset = None
    if config.get("eval_data_path"):
        try:
            eval_dataset = GPT2Dataset(
                data_path=config["eval_data_path"],
                use_memory_map=True
            )
        except FileNotFoundError:
            print(f"Warning: Evaluation data not found at {config['eval_data_path']}")
    
    # 最適化された学習の実行
    trainer.train_with_optimization(
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        batch_size=config["batch_size"],
        num_epochs=config["num_epochs"],
        use_dynamic_batching=config.get("use_dynamic_batching", False),
        max_tokens_per_batch=config.get("max_tokens_per_batch", None)
    )
    
    # 最終的なチェックポイントを保存
    trainer.save_checkpoint("final")
    
    print(f"\nOptimized training for phase {phase} completed.")
    print(f"Best model saved at {config['output_dir']}/checkpoint_best")


if __name__ == "__main__":
    # テスト用の設定
    test_config = {
        "description": "Test Optimized Training",
        "train_data_path": "data/phase1/training_sequences.pt",
        "output_dir": "checkpoints/test_optimized",
        "tokenizer_path": "tokenizer/japanese_bpe.model",
        "model_size": "small",
        "learning_rate": 2e-4,
        "num_epochs": 1,
        "batch_size": 2,
        "gradient_accumulation_steps": 16,
        "warmup_steps": 100,
        "use_gradient_checkpointing": True,
        "use_fused_optimizer": True,
        "compile_model": False
    }
    
    train_phase_optimized(1, test_config)