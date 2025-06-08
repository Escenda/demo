"""
demo3_optimized.py – Accelerated version of loop_gpt2.py

Applied speed‑up techniques:
1. Enable TF32 matmul & cuDNN
2. FlashAttention‑2 via transformers config
3. AMP (bfloat16) with GradScaler
4. torch.compile for kernel fusion / autotune
5. Fused AdamW optimiser
6. DataLoader tuned for overlap (num_workers, pin_memory, prefetch)
Notes: Requires PyTorch 2.2+ and transformers 4.40+ with flash‑attention‑2 support.
"""
from __future__ import annotations

import math, itertools, torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Global backend flags – free speed‑ups on Ampere/Hopper GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------
MODEL_NAME   = "gpt2"
MEM_TOKENS   = 16
LOOP_PE_SIZE = 64
BPTT_STEPS   = 4
GAMMA_INIT   = 0.01

tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def collate(batch, max_len=128):
    texts = [x["text"] for x in batch]
    toks = tokenizer(texts, truncation=True, max_length=max_len,
                     padding="max_length", return_tensors="pt")
    input_ids = toks["input_ids"]
    attn_mask = toks["attention_mask"]
    labels = input_ids.clone()
    labels[attn_mask == 0] = -100
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

class LoopGPT2(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        cfg = GPT2Config.from_pretrained(model_name)
        cfg.attn_implementation = "flash_attention_2"   # ✨
        self.gpt2 = GPT2Model.from_pretrained(model_name, config=cfg)
        self.hidden_dim  = cfg.n_embd
        self.vocab_size  = cfg.vocab_size

        # --- extra embeddings ------------------------------------------------
        self.mem_tok   = nn.Parameter(torch.randn(1, MEM_TOKENS, self.hidden_dim))
        self.loop_emb  = nn.Embedding(10_000, LOOP_PE_SIZE)
        self.loop_lin  = nn.Linear(LOOP_PE_SIZE, self.hidden_dim) if LOOP_PE_SIZE != self.hidden_dim else nn.Identity()

        # --- bridge ----------------------------------------------------------
        self.bridge    = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh())
        self.gamma     = nn.Parameter(torch.tensor(GAMMA_INIT), requires_grad=False)

        self.out_proj  = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self.out_proj.weight = self.gpt2.wte.weight  # weight tying

    # ------------- helpers ---------------------------------------------------
    def _apply_loop_pe(self, x: torch.Tensor, loop_step: int):
        B, T, _ = x.size()
        ids = torch.full((B, T), loop_step, device=x.device, dtype=torch.long)
        return x + self.loop_lin(self.loop_emb(ids))

    def _run_gpt2(self, inp_emb: torch.Tensor, attention_mask: torch.Tensor | None = None):
        return self.gpt2(inputs_embeds=inp_emb, attention_mask=attention_mask, use_cache=False).last_hidden_state

    # ------------- inference -------------------------------------------------
    @torch.no_grad()
    def stream(self, src_ids: torch.LongTensor, max_steps: int | None = None):
        self.eval()
        device = next(self.parameters()).device
        src_emb = self.gpt2.wte(src_ids.to(device)) + self.gpt2.wpe(torch.arange(src_ids.size(1), device=device))
        feedback  = torch.zeros(src_emb.size(0), 1, self.hidden_dim, device=device)
        mem_state = self.mem_tok.repeat(src_emb.size(0), 1, 1)
        loop_step = 0
        while max_steps is None or loop_step < max_steps:
            enc_in  = torch.cat([mem_state, src_emb, feedback], dim=1)
            enc_in  = self._apply_loop_pe(enc_in, loop_step)
            enc_out = self._run_gpt2(enc_in)

            dec_in  = torch.cat([src_emb, feedback], dim=1)
            dec_in  = self._apply_loop_pe(dec_in, loop_step)
            dec_out = self._run_gpt2(dec_in)

            h_t   = dec_out[:, -1]
            probs = F.softmax(self.out_proj(h_t), dim=-1)
            yield probs

            fb        = self.bridge(h_t).unsqueeze(1)
            feedback  = self.gamma * fb + (1 - self.gamma) * feedback
            mem_state = enc_out[:, :MEM_TOKENS, :]
            loop_step += 1

    # ------------- training --------------------------------------------------
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor,
                *, bptt_steps: int = BPTT_STEPS, kl_weight: float = 0.0):
        device = input_ids.device
        src_emb = self.gpt2.wte(input_ids) + self.gpt2.wpe(torch.arange(input_ids.size(1), device=device))
        feedback  = torch.zeros(src_emb.size(0), 1, self.hidden_dim, device=device)
        mem_state = self.mem_tok.repeat(src_emb.size(0), 1, 1)

        total_loss = torch.tensor(0.0, device=device)
        op_prev = None
        for loop_step in range(bptt_steps):
            enc_in  = torch.cat([mem_state, src_emb, feedback], dim=1)
            enc_in_mask = F.pad(attention_mask, (MEM_TOKENS, 1), value=1)
            enc_in  = self._apply_loop_pe(enc_in, loop_step)
            enc_out = self._run_gpt2(enc_in, attention_mask=enc_in_mask)

            dec_in  = torch.cat([src_emb, feedback], dim=1)
            dec_in_mask = F.pad(attention_mask, (0, 1), value=1)
            dec_in  = self._apply_loop_pe(dec_in, loop_step)
            dec_out = self._run_gpt2(dec_in, attention_mask=dec_in_mask)

            logits = self.out_proj(dec_out[:, :-1, :])
            loss_step = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100
            )

            if kl_weight and op_prev is not None:
                mask = labels != -100
                kl_loss = F.kl_div(F.log_softmax(logits[mask], dim=-1),
                                   op_prev[mask], reduction="batchmean", log_target=False)
                loss_step = loss_step + kl_weight * kl_loss

            total_loss = total_loss + loss_step
            op_prev = F.softmax(logits.detach(), dim=-1)

            h_t = F.layer_norm(dec_out.mean(dim=1), (self.hidden_dim,))
            feedback  = (self.gamma * self.bridge(h_t).unsqueeze(1) + (1 - self.gamma) * feedback).detach()
            mem_state = enc_out[:, :MEM_TOKENS, :].detach()

        return total_loss / bptt_steps

# ---------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_model = LoopGPT2().to(device)
    # model = torch.compile(raw_model, mode="max-autotune")  # torch 2.2+
    model = raw_model

    # freeze base GPT‑2 weights
    raw_model.gpt2.requires_grad_(False)
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, betas=(0.9, 0.98),
                                  weight_decay=0.01, fused=True)

    # -----------------------------------------------------------------
    print("--- Preparing Dataset ---")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds = ds.filter(lambda ex: len(ex["text"].split()) > 10)

    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate,
                        num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    print(f"Dataset size: {len(ds)}")

    # -----------------------------------------------------------------
    scaler = GradScaler()
    model.train()
    MAX_STEP = 3000
    GRAD_ACC = 4
    step = 0
    for epoch in itertools.count():
        for batch in loader:
            if step >= MAX_STEP:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(dtype=torch.bfloat16):
                loss = model(**batch, bptt_steps=BPTT_STEPS, kl_weight=0.0) / GRAD_ACC

            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACC == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if (step + 1) % (GRAD_ACC * 25) == 0:
                    print(f"Step {step+1}: loss={loss.item()*GRAD_ACC:.4f}")
            step += 1
        if step >= MAX_STEP:
            break
    print("Training done.")