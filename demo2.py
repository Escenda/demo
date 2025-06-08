"""
loop_transformer.py  ―  Encoder→Decoder→Encoder を閉ループで回す最小実装
  - PyTorch >= 2.0
  - 1GPU 16 GB で動作確認想定（小規模埋め込み・層数）
  - 学習: truncated BPTT（例では8ステップ）
  - 推論: 無停止ジェネレータ
"""

import itertools, math, torch, torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------
# ハイパーパラメータ（16 GB に収まる目安）
# ---------------------------------------------
VOCAB_SIZE   =   8192         # BPE でも可
D_MODEL      =    256
N_HEAD       =      4
N_LAYER      =      4         # Encoder/Decoder 共通で重み共有
MEM_TOKENS   =     16         # 内在的“長期記憶”用
MAX_CTX      =    128         # 入力列長
LOOP_PE_SIZE =     64         # ループ回数埋め込み次元
DROPOUT      =    0.1

# ---------------------------------------------
# 基本ブロック
# ---------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn  = nn.MultiheadAttention(D_MODEL, N_HEAD, dropout=DROPOUT, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(D_MODEL, 4*D_MODEL), nn.GELU(),
            nn.Linear(4*D_MODEL, D_MODEL), nn.Dropout(DROPOUT)
        )
        self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)

    def forward(self, x, attn_mask=None, kv_cache=None):
        # kv_cache: 推論高速化用 (重み共有なので reuse 可)
        k, v = (kv_cache or (None, None))
        y, _ = self.attn(x, k or x, v or x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + y)
        x = self.ln2(x + self.ff(x))
        return x

# ---------------------------------------------
# ループ Transformer
# ---------------------------------------------
class LoopTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # 共有 Encoder / Decoder
        self.shared_block = nn.ModuleList([TransformerBlock() for _ in range(N_LAYER)])

        # トークン・位置エンベディング
        self.tok_emb   = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb   = nn.Embedding(MAX_CTX + MEM_TOKENS, D_MODEL)
        self.loop_emb  = nn.Embedding(10_000, LOOP_PE_SIZE)  # ループ回数
        # learnable memory tokens
        self.mem_tok   = nn.Parameter(torch.randn(1, MEM_TOKENS, D_MODEL))

        # 出力ヘッド
        self.out_proj  = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        # Decoder → Encoder へのブリッジ
        self.bridge    = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.Tanh()
        )
        # フィードバック安定化用係数 γ（学習も可）
        self.gamma     = nn.Parameter(torch.tensor(0.1))

    # --- 1 ループ分 ---
    def _encoder(self, x, loop_step):
        # 位置 + ループ深さ埋め込み
        b, t, _ = x.size()
        pos_ids  = torch.arange(t, device=x.device).unsqueeze(0)
        loop_ids = torch.full((b, t), loop_step, device=x.device)
        x = x + self.pos_emb(pos_ids) + F.pad(self.loop_emb(loop_ids), (0, D_MODEL-LOOP_PE_SIZE))
        # 共有ブロック
        for blk in self.shared_block:
            x = blk(x)
        return x

    def _decoder(self, tgt, enc_out):
        # causal mask
        sz = tgt.size(1)
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=tgt.device), diagonal=1)
        y = tgt + self.pos_emb(torch.arange(sz, device=tgt.device).unsqueeze(0))
        # 共有ブロック
        for blk in self.shared_block:
            y = blk(y, attn_mask=mask)
        # cross-attention（単純化: enc_out をキー/バリューに追加 concat）
        y = y + enc_out[:, :y.size(1), :]
        return y

    # --- 推論モード：無停止ストリーム ---
    @torch.no_grad()
    def stream(self, src_ids, max_steps=None):
        device = next(self.parameters()).device
        src_emb = self.tok_emb(src_ids.to(device))
        feedback = torch.zeros_like(self.mem_tok[:, :1])   # 初期疑似トークン
        prev_mem = self.mem_tok.repeat(src_ids.size(0), 1, 1)
        loop_step = 0
        while (max_steps is None) or (loop_step < max_steps):
            # Encoder 入力を構築
            enc_in = torch.cat([prev_mem, src_emb, feedback], dim=1)
            enc_out = self._encoder(enc_in, loop_step)

            # Decoder: ここでは自己回帰マスク無しで全 token 一括再予測
            dec_out = self._decoder(src_emb, enc_out)
            h_t = dec_out[:, -1]                       # 末尾 token hidden
            logits = self.out_proj(h_t)
            probs = F.softmax(logits, dim=-1)

            yield probs                                 # >>> 外部へストリーム出力 <<<

            # 橋渡し & フィードバック更新
            fb = self.bridge(h_t)
            feedback = self.gamma * fb + (1. - self.gamma) * feedback
            # “メモリ”を更新（ここでは単純置換。加算/GRU 等も可）
            prev_mem = enc_out[:, :MEM_TOKENS, :]

            loop_step += 1

    # --- 学習モード（truncated BPTT 例） ---
    def forward(self, src_ids, tgt_ids, bptt_steps=8):
        device = next(self.parameters()).device
        src_emb = self.tok_emb(src_ids.to(device))
        tgt_emb = self.tok_emb(tgt_ids.to(device))

        feedback = torch.zeros_like(self.mem_tok[:, :1])
        prev_mem = self.mem_tok.repeat(src_ids.size(0), 1, 1)
        logits_all, loss_all = [], []

        for loop_step in range(bptt_steps):
            enc_in  = torch.cat([prev_mem, src_emb, feedback], dim=1)
            enc_out = self._encoder(enc_in, loop_step)
            dec_out = self._decoder(tgt_emb, enc_out)
            h_t     = dec_out[:, -1]
            logits  = self.out_proj(h_t)
            logits_all.append(logits)

            # 例: 各ループで “次トークン” を予測させる CE Loss
            loss_all.append(F.cross_entropy(logits, tgt_ids[:, 1]))

            feedback = self.gamma * self.bridge(h_t) + (1. - self.gamma) * feedback
            prev_mem = enc_out[:, :MEM_TOKENS, :]

        return torch.stack(loss_all).mean(), torch.stack(logits_all, dim=1)

# ---------------------------------------------
# 実行
# ---------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = LoopTransformer().to(device)

    # ダミーデータ
    BATCH = 2
    src = torch.randint(0, VOCAB_SIZE, (BATCH, MAX_CTX))
    tgt = torch.randint(0, VOCAB_SIZE, (BATCH, MAX_CTX))

    # --- 学習ステップ ---
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss, _ = model(src, tgt)
    loss.backward(); optim.step()
    print(f"Initial loss: {loss.item():.3f}")

    # --- ストリーム推論 ---
    gen = model.stream(src[:1], max_steps=5)
    for step, p in enumerate(gen):
        tok = torch.argmax(p, dim=-1).item()
        print(f"loop {step}: token={tok}")
