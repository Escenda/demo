import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import time

print("CUDA 利用可能:", torch.cuda.is_available())
print("使用中のデバイス:", torch.cuda.current_device())
print("GPU名:", torch.cuda.get_device_name(torch.cuda.current_device()))

# 比較する文章
sentence1 = "食事の際に細かな手の使用が困難なため、できるようになりたい。"
sentence2 = "食事動作ができるようになりたい。"


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"    {func.__name__} 実行時間: {end - start:.4f} 秒")
        return result

    return wrapper


@measure_time
def get_embedding(model, sentence):
    return model.encode(sentence, convert_to_tensor=True)


def test_model(model_name):
    print(f"\n=== {model_name} のテスト ===")
    
    # モデル初期化の時間測定
    start_time = time.time()
    model = SentenceTransformer(model_name, device="cuda")
    init_time = time.time() - start_time
    print(f"モデル初期化時間: {init_time:.4f} 秒")
    
    # ウォームアップ（空の処理）
    start_time = time.time()
    _ = model.encode("テスト", convert_to_tensor=True)
    warmup_time = time.time() - start_time
    print(f"ウォームアップ時間: {warmup_time:.4f} 秒")
    
    # 実際の測定（3回実行）
    print("実際の処理時間:")
    for i in range(3):
        print(f"  実行 {i+1}:")
        embeddings1 = get_embedding(model, sentence1)
        embeddings2 = get_embedding(model, sentence2)
    
    cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0]
    print(f"類似度: {cosine_score:.4f}")


# 各モデルでテスト
models = [
    "paraphrase-xlm-r-multilingual-v1",
    "paraphrase-multilingual-mpnet-base-v2", 
    "paraphrase-multilingual-MiniLM-L12-v2",
    "stsb-xlm-r-multilingual"
]

for model_name in models:
    test_model(model_name)