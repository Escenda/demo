# LLM訓練の高速化手法

このプロジェクトでは、LLMの訓練を高速化するための様々な最適化手法を実装しています。

## 実装した最適化手法

### 1. グラディエント・アキュムレーション（Gradient Accumulation）
- **実装箇所**: `train.py`, `train_optimized.py`
- **効果**: GPUメモリの制約下でも大きな実効バッチサイズを実現
- **設定**: `gradient_accumulation_steps=16`

### 2. 混合精度訓練（Mixed Precision Training / AMP）
- **実装箇所**: `train.py`, `train_optimized.py`
- **効果**: メモリ使用量を削減し、計算速度を向上
- **使用方法**: `fp16=True`でGradScalerとautocastを自動適用

### 3. グラディエント・チェックポインティング（Gradient Checkpointing）
- **実装箇所**: `model.py`の`TransformerBlock`
- **効果**: メモリ使用量を大幅に削減（計算時間とのトレードオフ）
- **使用方法**: `use_gradient_checkpointing=True`

### 4. 効率的なデータローディング
- **実装箇所**: `train.py`, `train_optimized.py`
- **最適化内容**:
  - メモリマップによるデータ読み込み
  - ピンメモリ（`pin_memory=True`）
  - 永続的ワーカー（`persistent_workers=True`）
  - プリフェッチ（`prefetch_factor=2`）
  - 適切なワーカー数の設定

### 5. メモリ最適化
- **実装箇所**: `train_optimized.py`
- **最適化内容**:
  - `zero_grad(set_to_none=True)`による効率的な勾配リセット
  - 定期的なガベージコレクションとキャッシュクリア
  - メモリ使用量のモニタリング

### 6. Fused Optimizer
- **実装箇所**: `train_optimized.py`
- **効果**: 複数の演算を融合して高速化
- **使用方法**: `AdamW`の`fused=True`オプション

### 7. モデルコンパイル（PyTorch 2.0+）
- **実装箇所**: `train_optimized.py`
- **効果**: JITコンパイルによる実行速度の向上
- **使用方法**: `compile_model=True`（PyTorch 2.0以降）

### 8. 学習率スケジューリング
- **実装箇所**: `train.py`, `train_optimized.py`
- **手法**: 
  - 線形ウォームアップ
  - コサイン減衰
  - OneCycleLR（最適化版）

## 使用方法

### 通常の訓練
```bash
python main.py train
```

### 最適化を無効にした訓練
```bash
python main.py train --no-optimize
```

### 特定フェーズのみの訓練
```bash
python main.py train --phase 1
```

### ベンチマークの実行
```bash
python benchmark.py --all --model-size small
```

## パフォーマンス比較

ベンチマークツールを使用して、各最適化の効果を測定できます：

```bash
# すべてのベンチマークを実行
python benchmark.py --all

# 結果は benchmark_results/ に保存されます
```

## メモリ使用量の目安

| モデルサイズ | 通常 | 最適化済み | 削減率 |
|------------|------|-----------|--------|
| Small (117M) | ~4GB | ~2.5GB | ~38% |
| Medium (345M) | ~8GB | ~5GB | ~38% |
| Large (774M) | ~16GB | ~10GB | ~38% |

## 推奨設定

### GPUメモリが限られている場合（8GB以下）
```python
config = {
    "use_gradient_checkpointing": True,
    "fp16": True,
    "gradient_accumulation_steps": 32,
    "batch_size": 1,
    "use_fused_optimizer": True
}
```

### 高速訓練を優先する場合
```python
config = {
    "use_gradient_checkpointing": False,
    "fp16": True,
    "gradient_accumulation_steps": 8,
    "batch_size": 4,
    "use_fused_optimizer": True,
    "compile_model": True  # PyTorch 2.0+
}
```

## トラブルシューティング

### OOMエラーが発生する場合
1. `batch_size`を減らす
2. `gradient_accumulation_steps`を増やす
3. `use_gradient_checkpointing=True`を設定
4. モデルサイズを小さくする

### 訓練が遅い場合
1. `num_workers`を増やす（CPU数に応じて）
2. `pin_memory=True`を確認
3. `compile_model=True`を試す（PyTorch 2.0+）
4. データをSSDに配置する

### 精度が低下する場合
1. 学習率を調整する（混合精度訓練時は特に重要）
2. ウォームアップステップを増やす
3. 勾配クリッピングの値を調整する