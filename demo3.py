"""
loop_gpt2.py  ―  GPT-2 ベースで Encoder→Decoder→Encoder を閉ループで回す最小実装
  * transformers==4.40 で動作確認
  * 16 GB GPU で gpt2-small（124M）を想定
  * Universal 重み共有・メモリトークン・ループ PE・橋渡しフィードバック
  * 学習: truncated BPTT (例では 4 step)。
  * 推論: model.stream(...) が無停止で確率分布を yield。

元ファイル demo2.py (loop_transformer.py) を GPT-2 に置き換えて全面改修。
"""
from __future__ import annotations

import itertools, math, torch, torch.nn as nn, torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader

# ------------------------- ハイパーパラメータ -------------------------
MODEL_NAME   = "gpt2"          # gpt2-small 124M
MEM_TOKENS   = 16              # 内在的 "長期記憶"
LOOP_PE_SIZE = 64              # ループ回数埋め込みベクトル
BPTT_STEPS   = 4               # truncated BPTT 反復数
GAMMA_INIT   = 0.01            # フィードバック混合率 γ の初期値 (固定)

# ------------------------- 主要モジュール ----------------------------
class LoopGPT2(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        cfg              = GPT2Config.from_pretrained(model_name)
        # cross_attention=False なので causal mask のまま利用
        self.gpt2        = GPT2Model.from_pretrained(model_name, config=cfg)
        self.hidden_dim  = cfg.n_embd
        self.vocab_size  = cfg.vocab_size

        # ------- 拡張エンベディング -------
        # 学習可能メモリトークン (1, K, D)
        self.mem_tok   = nn.Parameter(torch.randn(1, MEM_TOKENS, self.hidden_dim))
        # ループ回数埋め込み
        self.loop_emb  = nn.Embedding(10_000, LOOP_PE_SIZE)
        # ループ PE を GPT-2 hidden 次元に合わせる線形層
        self.loop_lin: nn.Module
        if LOOP_PE_SIZE != self.hidden_dim:
            self.loop_lin = nn.Linear(LOOP_PE_SIZE, self.hidden_dim)
        else:
            self.loop_lin = nn.Identity()

        # Decoder → Encoder への橋渡し
        self.bridge    = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )
        # γ: 学習させず固定
        self.gamma     = nn.Parameter(torch.tensor(GAMMA_INIT), requires_grad=False)

        # 出力ヘッド (GPT-2 wte と weight tie)
        self.out_proj  = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self.out_proj.weight = self.gpt2.wte.weight  # weight tying

    # ----------------------------- ヘルパ -----------------------------
    def _apply_loop_pe(self, x: torch.Tensor, loop_step: int):
        """x: (B, T, D) にループ埋め込みを加算"""
        B, T, _ = x.size()
        ids     = torch.full((B, T), loop_step, device=x.device, dtype=torch.long)
        pe      = self.loop_lin(self.loop_emb(ids))
        return x + pe

    # --------------------------- 1 ループ分 ---------------------------
    def _run_gpt2(self, inp_emb: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """GPT-2 を埋め込み直接入力で呼び出し、hidden を返す"""
        return self.gpt2(inputs_embeds=inp_emb, attention_mask=attention_mask, use_cache=False).last_hidden_state

    # --------------------------- 推論 (無停止) ------------------------
    @torch.no_grad()
    def stream(self, src_ids: torch.LongTensor, max_steps: int | None = None):
        """src_ids: (1, T) など。yield で確率分布を返し続ける。"""
        self.eval()
        device = next(self.parameters()).device
        src_emb = self.gpt2.wte(src_ids.to(device)) + self.gpt2.wpe(torch.arange(src_ids.size(1), device=device))

        # 初期フィードバックおよびメモリ
        feedback  = torch.zeros(src_emb.size(0), 1, self.hidden_dim, device=device)
        mem_state = self.mem_tok.repeat(src_emb.size(0), 1, 1)

        loop_step = 0
        while (max_steps is None) or (loop_step < max_steps):
            # ----- Encoder フェーズ -----
            enc_in  = torch.cat([mem_state, src_emb, feedback], dim=1)
            enc_in  = self._apply_loop_pe(enc_in, loop_step)
            enc_out = self._run_gpt2(enc_in)

            # ----- Decoder フェーズ ----- (ここでは enc_out の末尾 token を query)
            dec_query = src_emb  # 最小実装：同じ入力列をデコード観点として使う
            dec_in    = torch.cat([dec_query, feedback], dim=1)
            dec_in    = self._apply_loop_pe(dec_in, loop_step)
            dec_out   = self._run_gpt2(dec_in)

            h_t   = dec_out[:, -1]                      # 最終 token hidden
            logits = self.out_proj(h_t)
            probs  = F.softmax(logits, dim=-1)
            yield probs                                 # ---- 外部へ出力 ----

            # ----- ブリッジ & メモリ更新 -----
            fb        = self.bridge(h_t).unsqueeze(1)   # (B,1,D)
            feedback  = self.gamma * fb + (1 - self.gamma) * feedback
            mem_state = enc_out[:, :MEM_TOKENS, :]      # 最新 Encoder 出力を mem に
            loop_step += 1

    # --------------------------- 学習 (trunc BPTT) -------------------
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor, *, bptt_steps: int = BPTT_STEPS, kl_weight: float = 0.0):
        """各ループで次トークン予測。loss を返す。"""
        device   = next(self.parameters()).device
        input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)

        src_emb  = self.gpt2.wte(input_ids) + self.gpt2.wpe(torch.arange(input_ids.size(1), device=device))

        feedback  = torch.zeros(src_emb.size(0), 1, self.hidden_dim, device=device)
        mem_state = self.mem_tok.repeat(src_emb.size(0), 1, 1)

        total_loss = torch.tensor(0.0, device=device)
        op_prev = None

        for loop_step in range(bptt_steps):
            # Encoder: メモリ、入力、フィードバックを結合
            enc_in  = torch.cat([mem_state, src_emb, feedback], dim=1)
            enc_in_mask = F.pad(attention_mask, (MEM_TOKENS, 1), 'constant', 1)
            enc_in  = self._apply_loop_pe(enc_in, loop_step)
            enc_out = self._run_gpt2(enc_in, attention_mask=enc_in_mask)

            # Decoder: 入力とフィードバックを結合
            dec_in  = torch.cat([src_emb, feedback], dim=1)
            dec_in_mask = F.pad(attention_mask, (0, 1), 'constant', 1)
            dec_in  = self._apply_loop_pe(dec_in, loop_step)
            dec_out = self._run_gpt2(dec_in, attention_mask=dec_in_mask)

            # --- 損失計算 ---
            logits = self.out_proj(dec_out[:, :-1, :])
            loss_predict = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100
            )
            loss_step = loss_predict

            # --- KL正則化 ---
            op_t = F.softmax(logits, dim=-1)
            if kl_weight > 0.0 and op_prev is not None:
                mask = labels != -100
                
                # マスクを適用して実トークンのみを対象にする
                op_t_masked = op_t[mask]
                op_prev_masked = op_prev[mask]
                
                if op_t_masked.numel() > 0: # 実トークンが存在する場合のみ計算
                    kl_loss = F.kl_div(op_t_masked.log(), op_prev_masked, reduction="batchmean", log_target=False)
                    loss_step = loss_step + kl_weight * kl_loss

            total_loss = total_loss + loss_step
            op_prev = op_t.detach()

            # --- フィードバックとメモリの更新 ---
            h_t = F.layer_norm(dec_out.mean(dim=1), (self.hidden_dim,))
            feedback  = (self.gamma * self.bridge(h_t).unsqueeze(1) + (1 - self.gamma) * feedback).detach()
            mem_state = enc_out[:, :MEM_TOKENS, :].detach()

        return total_loss / bptt_steps

# ------------------------------- テスト --------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = LoopGPT2().to(device)
    tok    = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token

    # --- 1. 学習対象とオプティマイザ設定 ---
    model.gpt2.requires_grad_(False)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=1e-5, betas=(0.9, 0.98), weight_decay=0.01)
    
    print("--- Trainable parameters ---")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"- {name}")

    # --- 2. データセットとデータローダーの準備 ---
    print("\n--- Preparing Dataset (filtering short lines) ---")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds = ds.filter(lambda ex: len(ex["text"].strip().split()) > 10) # 短すぎる行を除外

    def collate_fn(batch, max_len=128):
        texts = [b["text"] for b in batch]
        toks = tok(
            texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt")

        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100 # PAD部分は-100に
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn, shuffle=True)
    print(f"Dataset prepared with {len(ds)} samples.")

    # --- 3. ウォームアップ学習ループ ---
    print("\n--- Warm-up Training ---")
    model.train()
    
    MAX_STEPS = 6000
    GRAD_ACCUM_STEPS = 4
    step = 0
    
    for epoch in range(20):
        for i, batch in enumerate(loader):
            if step >= MAX_STEPS: break
            if batch is None: continue
            
            # KL正則化の重みを線形ウォームアップ
            if step < 3000:
                kl_weight = 0.0
            elif step < 6000:
                kl_weight = 0.05 * (step - 3000) / 3000
            else:
                kl_weight = 0.05 + 0.1 * min((step - 6000) / 6000, 1.0)


            loss = model(
                input_ids=batch['input_ids'],
                labels=batch['labels'],
                attention_mask=batch['attention_mask'],
                kl_weight=kl_weight,
                bptt_steps=BPTT_STEPS
            )
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()
                
                # 勾配累積のステップを1ステップとしてカウント
                actual_step = step // GRAD_ACCUM_STEPS
                if (actual_step + 1) % 25 == 0: # 100バッチごと（400イテレーションごと）
                    print(f"Step {actual_step+1}/{MAX_STEPS//GRAD_ACCUM_STEPS}, Loss: {loss.item() * GRAD_ACCUM_STEPS:.4f}, KL_w: {kl_weight:.4f}")
                step += 1

        if step >= MAX_STEPS: break
            
    print("Warm-up training finished.")

    # --- 4. 推論ストリーム (学習後) ---
    print("\n--- Inference Stream (after training) ---")
    model.eval()
    prompt = "In a shocking finding, scientist discovered"
    src = tok(prompt, return_tensors="pt").input_ids
    
    gen = model.stream(src, max_steps=15)
    generated_tokens = []
    current_tokens = src.tolist()[0]

    for i, p in enumerate(gen):
        p[:, tok.eos_token_id] = -float('Inf')
        
        # Top-k sampling
        top_k = 40
        top_k_probs, top_k_indices = torch.topk(p, top_k)
        sampled_index = torch.multinomial(F.softmax(top_k_probs, dim=-1), 1)
        tok_id = torch.gather(top_k_indices, 1, sampled_index).item()
        
        token_str = tok.decode([tok_id])
        generated_tokens.append(token_str)
        print(f"loop {i}: token = {token_str}")

    print("\nGenerated sequence:")
    print(prompt + "".join(generated_tokens))
