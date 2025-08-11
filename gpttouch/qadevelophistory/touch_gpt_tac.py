import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import httpx
import random
import numpy as np
import pandas as pd
import faiss
import torch
from tqdm import tqdm
from imagebind.models.imagebind_model import imagebind_huge, ModalityType
from imagebind.data import load_and_transform_vision_data
from openai import OpenAI  # 新版 SDK 推荐方式
import base64

# 设置 HTTP 和 HTTPS 代理（按需修改端口和地址）
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
# import openai
# --------- 配置区域 ---------
device = "cuda" if torch.cuda.is_available() else "cpu"

EMB_ROOT = "embeddings/embeddings_tac"
TRAIN_CSV = "data/ssvtp/new_train.csv"  # 使用新版本 CSV
PREFIX_FILE = "../data/ssvtp/text_prefix.txt"

EMB_DIM = 1024
TOP_K = 3

# print("httpx loaded from:", httpx.__file__)

# 构造 httpx 客户端用于代理

proxy_url = "http://127.0.0.1:7890"
http_client = httpx.Client()

# client = OpenAI(api_key="...", http_client=http_client)
client = OpenAI(
    api_key="sk-proj-",
    http_client=http_client
)
# client = OpenAI(api_key="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")
# client = OpenAI(
#     api_key="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"
# )
# openai.api_key ="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"
MODEL_NAME = "gpt-3.5-turbo"

# --------- 1. 加载 ImageBind 模型 ---------
print("Loading ImageBind model …")
model = imagebind_huge(pretrained=True).to(device)
model.eval()

# --------- 2. 加载 embeddings 和索引 ---------
print("Loading embeddings index …")
all_emb = np.load(os.path.join(EMB_ROOT, "all_embeddings.npy"))  # (N, D)
all_idx = np.load(os.path.join(EMB_ROOT, "all_indices.npy"))  # (N,)

# 归一化
emb_norm = all_emb / np.linalg.norm(all_emb, axis=1, keepdims=True)
index = faiss.IndexFlatIP(EMB_DIM)
index.add(emb_norm.astype('float32'))

# --------- 3. 加载新 CSV ---------
df = pd.read_csv(TRAIN_CSV, dtype={"index": str})
# 包含列 ['url', 'tactile', 'caption', 'index']

# --------- 4. 加载 prefix 前缀 ---------
with open(PREFIX_FILE, encoding="utf-8") as f:
    prefixes = [line.strip() for line in f if line.strip()]


# --------- 5. 检索并生成 Prompt ---------
def generate_prompt_for_image(image_path: str) -> str:
    vis = load_and_transform_vision_data([image_path], device)
    with torch.no_grad():
        out = model({ModalityType.VISION: vis})[ModalityType.VISION]
    emb = out.cpu().numpy()[0]
    emb = emb / np.linalg.norm(emb)

    D, I = index.search(emb.reshape(1, -1).astype('float32'), TOP_K)
    inds = I[0]

    neighbors = []
    for pos in inds:
        idx = all_idx[pos]
        row = df[df["index"] == idx]
        if not row.empty:
            neighbors.append(row.iloc[0]["caption"])
        else:
            neighbors.append("<unknown>")

    prefix = random.choice(prefixes)
    # prompt = f"{prefix}{', '.join(neighbors)}"
    prompt = f"{prefix} This image resembles the following textures: {', '.join(neighbors)}."

    return prompt

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# --------- 6. 示例调用 ---------
if __name__ == "__main__":
    test_image = "data/ssvtp/images_tac/image_123_tac.jpg"
    prompt = generate_prompt_for_image(test_image)

    print(">> Generated Prompt:", prompt)

    # 可选：调用 OpenAI GPT 生成回复
    # 使用新版 OpenAI SDK 发送请求
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a tactile-description assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=64,
    )
    # response = openai.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a tactile-description assistant."},
    #         {"role": "user", "content": prompt},
    #     ],
    #     temperature=0.7,
    #     max_tokens=64,
    # )
    print(">> GPT Response:", response.choices[0].message.content)
