import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from model import get_gpt2_model
from tokenizer import JapaneseTokenizer
from transformers import get_linear_schedule_with_warmup


class GPT2Dataset(Dataset):
    """GPT-2学習用データセット"""
    
    def __init__(self, data_path: str, max_length: int = 1024):
        self.data_path = data_path
        self.max_length = max_length
        
        # データを読み込む
        if data_path.endswith('.pt'):
            # 事前にトークナイズされたテンソル
            self.data = torch.load(data_path)
        else:
            # JSONLファイルから読み込み
            self.data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    self.data.append(entry)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(self.data, torch.Tensor):
            # 既にトークナイズ済み
            tokens = self.data[idx]
            return {
                "input_ids": tokens,
                "labels": tokens
            }
        else:
            # 生のテキストデータ（実際の使用時はトークナイザーが必要）
            return self.data[idx]


class Trainer:
    """GPT-2モデルのトレーナー"""
    
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
        max_grad_norm=1.0
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.max_grad_norm = max_grad_norm
        
        # オプティマイザの設定
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Mixed Precision用のスケーラー
        self.scaler = GradScaler() if fp16 else None
        
        # 学習統計
        self.global_step = 0
        self.epoch = 0
        
        os.makedirs(output_dir, exist_ok=True)
    
    def get_lr(self):
        """学習率スケジューリング（線形ウォームアップ + コサイン減衰）"""
        if self.global_step < self.warmup_steps:
            # ウォームアップ
            return self.learning_rate * self.global_step / self.warmup_steps
        else:
            # コサイン減衰
            progress = (self.global_step - self.warmup_steps) / (100000 - self.warmup_steps)
            return self.learning_rate * (0.5 * (1.0 + np.cos(progress * np.pi)))
    
    def train_epoch(self, train_loader, epoch):
        """1エポックの学習"""
        self.model.train()
        total_loss = 0
        step_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        
        for i, batch in enumerate(progress_bar):
            # バッチデータをデバイスに移動
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 勾配をリセット
            self.optimizer.zero_grad()
            
            # 順伝播
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
                    
                    # 学習率の更新
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.get_lr()
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # 学習率の更新
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.get_lr()
                    
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # ロスを記録
                total_loss += step_loss
                progress_bar.set_postfix({
                    'loss': f'{step_loss:.4f}',
                    'lr': f'{self.get_lr():.2e}',
                    'step': self.global_step
                })
                step_loss = 0
                
                # 定期的にチェックポイントを保存
                if self.global_step % 5000 == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
        
        return total_loss / len(train_loader)
    
    def evaluate(self, eval_loader):
        """評価"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids, labels=labels)
                loss = outputs["loss"]
                
                total_loss += loss.item() * input_ids.size(0)
                total_tokens += input_ids.numel()
        
        # Perplexityを計算
        avg_loss = total_loss / len(eval_loader)
        perplexity = np.exp(avg_loss)
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        batch_size=2,
        num_epochs=1,
        save_steps=5000,
        eval_steps=5000
    ):
        """学習のメインループ"""
        # DataLoaderの作成
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        eval_loader = None
        if eval_dataset:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # 学習開始
        print(f"Training started with {len(train_dataset)} samples")
        print(f"Batch size: {batch_size}, Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * self.gradient_accumulation_steps}")
        
        best_eval_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # 学習
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 評価
            if eval_loader and (epoch + 1) % 1 == 0:
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
            
            # エポックごとにチェックポイントを保存
            self.save_checkpoint(f"epoch_{epoch}")
            
            elapsed_time = time.time() - start_time
            print(f"Epoch completed in {elapsed_time:.2f} seconds")
    
    def save_checkpoint(self, name):
        """チェックポイントの保存"""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_{name}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # モデルの保存
        torch.save(
            self.model.state_dict(),
            os.path.join(checkpoint_path, "model.pt")
        )
        
        # オプティマイザの保存
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(checkpoint_path, "optimizer.pt")
        )
        
        # トレーニング状態の保存
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": getattr(self, 'best_eval_loss', float('inf'))
        }
        
        with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
            json.dump(state, f)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """チェックポイントの読み込み"""
        # モデルの読み込み
        self.model.load_state_dict(
            torch.load(os.path.join(checkpoint_path, "model.pt"))
        )
        
        # オプティマイザの読み込み
        self.optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint_path, "optimizer.pt"))
        )
        
        # トレーニング状態の読み込み
        with open(os.path.join(checkpoint_path, "training_state.json"), "r") as f:
            state = json.load(f)
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
        
        print(f"Checkpoint loaded: {checkpoint_path}")


def train_phase(phase, config):
    """特定のフェーズの学習を実行"""
    print(f"\n{'='*50}")
    print(f"Training Phase {phase}: {config['description']}")
    print(f"{'='*50}\n")
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # トークナイザの読み込み
    tokenizer = JapaneseTokenizer()
    if os.path.exists(config.get("tokenizer_path", "tokenizer/japanese_bpe.model")):
        tokenizer.load(config["tokenizer_path"])
    else:
        print("Warning: Tokenizer not found. Please train tokenizer first.")
        return
    
    # モデルの作成または読み込み
    model = get_gpt2_model(
        model_size=config.get("model_size", "medium"),
        vocab_size=len(tokenizer)
    )
    
    # 前のフェーズのチェックポイントを読み込む
    if config.get("previous_checkpoint"):
        print(f"Loading model from previous checkpoint: {config['previous_checkpoint']}")
        model.load_state_dict(torch.load(os.path.join(config["previous_checkpoint"], "model.pt")))
    
    model = model.to(device)
    
    print(f"  Training data path: {config['train_data_path']}")
    print(f"  Tokenizer path: {config['tokenizer_path']}")
    print(f"  Output directory: {config['output_dir']}")
    print(f"  Model size: {config.get('model_size', 'medium')}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Num epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Warmup steps: {config['warmup_steps']}")
    print("----------------------------------------")
    
    # データセットの準備
    try:
        # GPT2Datasetを使用してデータセットを読み込む
        dataset = GPT2Dataset(data_path=config["train_data_path"])
    except FileNotFoundError:
        print(f"Error: Training data not found at {config['train_data_path']}")
        print("Please generate the data for this phase first.")
        return
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # トレーナーの作成
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        fp16=config.get("fp16", True),
        max_grad_norm=config.get("max_grad_norm", 1.0)
    )
    
    # 評価データセットの準備 (もしあれば)
    eval_dataset = None
    if config.get("eval_data_path"):
        try:
            eval_dataset = GPT2Dataset(data_path=config["eval_data_path"])
        except FileNotFoundError:
            print(f"Warning: Evaluation data not found at {config['eval_data_path']}")

    # 学習の実行
    trainer.train(
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        batch_size=config["batch_size"],
        num_epochs=config["num_epochs"]
    )
    
    # 最終的なチェックポイントを保存
    trainer.save_checkpoint("best")
    
    print(f"\nTraining for phase {phase} completed.")
    print(f"Best model saved at {config['output_dir']}/checkpoint_best")


def main(configs):
    """4フェーズの学習を順番に実行"""
    
    # 4フェーズの学習を順番に実行
    for i in range(1, 5):
        phase_str = f"phase{i}"
        if phase_str in configs:
            config = configs[phase_str]
            config["phase"] = i # フェーズ番号を追加
            train_phase(i, config)
        else:
            print(f"Configuration for phase {i} not found. Skipping.")


if __name__ == "__main__":
    # このスクリプトを直接実行する場合のフォールバック
    # main.pyから呼び出されることを想定しているため、通常はここは使われない
    print("This script is intended to be run from main.py")
    print("Running with default standalone settings (not recommended)")
    
    # IȃftHgݒ (main.py ̋Lq<y_bin_556>)
    STANDALONE_CONFIGS = {
        "phase1": {
            "description": "Knowledge Explanation Learning",
            "train_data_path": "data/phase1/training_sequences.pt",
            "output_dir": "checkpoints/phase1",
            "tokenizer_path": "tokenizer/japanese_bpe.model",
            "model_size": "medium",
            "learning_rate": 2e-4,
            "num_epochs": 1,
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
            "num_epochs": 2,
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
            "num_epochs": 3,
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
            "num_epochs": 5,
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "warmup_steps": 200
        }
    }
    main(STANDALONE_CONFIGS)