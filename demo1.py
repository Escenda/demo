import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 再現性のために乱数シードを固定
np.random.seed(0)
torch.manual_seed(0)

# サンプルのテキストコーパス（パブリックドメインからの抜粋）
text = (
    "Alice was beginning to get very tired of sitting by her sister on the bank, "
    "and of having nothing to do: once or twice she had peeped into the book her sister was reading, "
    "but it had no pictures or conversations in it, "
    "'and what is the use of a book,' thought Alice 'without pictures or conversations?'"
)
# テキストから文字単位の語彙を作成する
chars = sorted(list(set(text)))
vocab_size = len(chars)
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}

# Transformerブロックの定義（セルフアテンション＋フィードフォワード）
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        # マルチヘッド・セルフアテンション層
        self.attn = nn.MultiheadAttention(n_embd, num_heads=n_head, dropout=dropout, batch_first=True)
        # フィードフォワードネットワーク（最後にドロップアウトを適用）
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        # 層正規化 (Layer Normalization)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # アテンション出力へのドロップアウト
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        # Self-Attention部: xは[batch, seq, embd]（batch_first=Trueなのでこの形）
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        # アテンション部の残差接続と正規化
        x = self.ln1(x + self.dropout(attn_output))
        # フィードフォワード部の残差接続と正規化
        ff_output = self.ff(x)
        x = self.ln2(x + ff_output)
        return x

# Transformerベースの言語モデル（GPT風）
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=2, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        # トークン埋め込み層と位置埋め込み層
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_seq_len, n_embd)
        # Transformerブロックのスタック
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        # 最終層のLayerNormと出力の全結合層
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx):
        # idx: [バッチサイズ, シーケンス長] のトークンIDテンソル
        b, t = idx.shape
        assert t <= self.max_seq_len, "Sequence length exceeds model's max_seq_len"
        # トークン埋め込みと位置埋め込みを加算
        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)  # shape [1, t]
        x = self.token_emb(idx) + self.pos_emb(pos)
        # 将来のトークンを参照しないためのマスクを作成（サイズ [t, t]）
        mask = torch.triu(torch.full((t, t), float('-inf'), device=idx.device), diagonal=1)
        # 各Transformerブロックを適用
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        # 最後にLayerNormを適用し、出力としてロジットを計算
        x = self.ln_f(x)
        logits = self.head(x)  # [batch, seq_len, vocab_size]
        return logits

# テキスト生成（推論）関数
def generate_text(model, start_text, length=100, device='cpu'):
    model.eval()  # 推論モードに切り替え
    # 開始テキストをインデックス列にエンコード
    input_indices = [char2idx[ch] for ch in start_text]
    input_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)
    generated_text = start_text
    for _ in range(length):
        # 長すぎる場合は直近のmax_seq_lenトークンに入力を切り詰め
        if input_tensor.size(1) > model.max_seq_len:
            input_tensor = input_tensor[:, -model.max_seq_len:]
        with torch.no_grad():
            logits = model(input_tensor)
        # 最後のトークンに対するロジットを取得
        last_logits = logits[0, -1, :]
        # 確率分布に変換（softmax）
        probs = F.softmax(last_logits, dim=-1)
        # 最も確率の高いトークンを選択（貪欲法）
        next_idx = torch.argmax(probs).item()
        # ランダムサンプリングする場合: next_idx = torch.multinomial(probs, num_samples=1).item()
        # 予測した文字を出力テキストに追加
        next_char = idx2char[next_idx]
        generated_text += next_char
        # 新しいトークンIDを入力テンソルに追加
        next_idx_tensor = torch.tensor([[next_idx]], device=device)
        input_tensor = torch.cat([input_tensor, next_idx_tensor], dim=1)
    return generated_text

# メイン処理（学習とデモ）
if __name__ == "__main__":
    # データセットをインデックスの列に変換して準備
    data = np.array([char2idx[ch] for ch in text], dtype=np.int64)
    data_size = len(data)
    # モデルのハイパーパラメータ
    embed_size = 64
    num_heads = 4
    num_layers = 2
    context_len = 64  # 学習時のシーケンス長（コンテキスト長）
    batch_size = 4
    dropout_rate = 0.1
    learning_rate = 1e-3
    num_iterations = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # モデルを初期化
    model = TransformerLanguageModel(vocab_size, n_embd=embed_size, n_head=num_heads,
                                     n_layer=num_layers, max_seq_len=context_len, dropout=dropout_rate).to(device)
    # 学習の設定（オプティマイザ・損失関数など）
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()
    # 学習ループ
    for step in range(num_iterations):
        # ランダムな開始位置からバッチをサンプリング
        batch_idx = np.random.randint(0, data_size - context_len - 1, size=batch_size)
        batch_inputs = []
        batch_targets = []
        for idx in batch_idx:
            batch_inputs.append(data[idx: idx + context_len])
            batch_targets.append(data[idx + 1: idx + context_len + 1])
        batch_inputs = torch.tensor(batch_inputs, dtype=torch.long, device=device)
        batch_targets = torch.tensor(batch_targets, dtype=torch.long, device=device)
        # フォワードパス（順伝播）
        logits = model(batch_inputs)
        # 損失を計算
        loss = criterion(logits.view(-1, vocab_size), batch_targets.view(-1))
        # 逆伝播してモデルを更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 一定間隔で損失を表示
        if step % 100 == 0:
            print(f"Step {step}/{num_iterations}: Loss = {loss.item():.4f}")
    # 学習後、生成をテスト
    prompt = "Alice was "
    output = generate_text(model, prompt, length=100, device=device)
    print("Prompt:", prompt)
    print("Generated continuation:", output)
