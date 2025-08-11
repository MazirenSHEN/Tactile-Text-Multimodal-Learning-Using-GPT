import os
import base64
import joblib
import torch
import numpy as np
from PIL import Image
from openai import OpenAI
from torchvision import transforms
from imagebind.models.imagebind_model import imagebind_huge, ModalityType
from imagebind.data import load_and_transform_vision_data

# ----------- 路径和模型加载 ------------
device = "cuda" if torch.cuda.is_available() else "cpu"
client = OpenAI(api_key="sk-proj-")  # 推荐从环境变量读取

classifier = joblib.load("../tactile_rgb_classifier.pkl")
model = imagebind_huge(pretrained=True).to(device).eval()

# 用于 RGB 图像的预处理
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_rgb_prompt(image_path: str, question: str) -> list:
    base64_image = encode_image_to_base64(image_path)
    return [
        {"type": "text", "text": question},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "low"
            }
        }
    ]

# 用于 tactile 图像的 prompt 生成（引用你的原逻辑）
import pandas as pd
import random
import faiss

EMB_ROOT = "embeddings/embeddings_tac"
TRAIN_CSV = "data/ssvtp/new_train.csv"
PREFIX_FILE = "../data/ssvtp/text_prefix.txt"
EMB_DIM = 1024
TOP_K = 3

# 加载索引和参考数据
all_emb = np.load(os.path.join(EMB_ROOT, "all_embeddings.npy"))
all_idx = np.load(os.path.join(EMB_ROOT, "all_indices.npy"))
emb_norm = all_emb / np.linalg.norm(all_emb, axis=1, keepdims=True)
index = faiss.IndexFlatIP(EMB_DIM)
index.add(emb_norm.astype("float32"))

df = pd.read_csv(TRAIN_CSV, dtype={"index": str})
with open(PREFIX_FILE, encoding="utf-8") as f:
    prefixes = [line.strip() for line in f if line.strip()]

# ---------- 多模态分流处理函数 ----------
def handle_image(image_path: str):
    # Step 1: 提取 embedding
    vis = load_and_transform_vision_data([image_path], device)
    with torch.no_grad():
        emb = model({ModalityType.VISION: vis})[ModalityType.VISION]
    emb_np = emb.cpu().numpy()

    # Step 2: 判断 RGB / Tactile
    pred = classifier.predict(emb_np)[0]
    is_rgb = (pred == 0)

    # Step 3: 路由逻辑
    if is_rgb:
        print("Detected RGB image, GPT")
        messages = [
            {"role": "system", "content": "You are a vision assistant that describes object properties."},
            {"role": "user", "content": build_rgb_prompt(image_path, "Please describe the texture and surface features of the objects in the image.")}
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=256,
            temperature=0.7,
        )
        print("GPT-4o：\n", response.choices[0].message.content)
    else:
        print("Detected Tactile image，use ImageBind + GPT ")
        emb_vec = emb_np[0] / np.linalg.norm(emb_np[0])
        D, I = index.search(emb_vec.reshape(1, -1).astype("float32"), TOP_K)
        neighbors = []
        for pos in I[0]:
            idx = all_idx[pos]
            row = df[df["index"] == idx]
            neighbors.append(row.iloc[0]["caption"] if not row.empty else "<unknown>")
        prefix = random.choice(prefixes)
        prompt = f"{prefix} This image resembles the following textures: {', '.join(neighbors)}."

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a tactile-description assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        print("GPT-4o：\n", response.choices[0].message.content)

# ---------- 运行入口 ----------
if __name__ == "__main__":
    test_image = "data/test_tac_1.jpg"  # 或 RGB 图像路径
    handle_image(test_image)
