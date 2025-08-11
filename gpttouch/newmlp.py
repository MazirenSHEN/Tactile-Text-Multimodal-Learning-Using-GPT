# tactile_text_rgb_align_from_csv.py

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import faiss
import clip
from PIL import Image
import pandas as pd
from sklearn.metrics import average_precision_score

# ========= 1. 数据加载（从CSV） =========

def load_from_csv(csv_path, tactile_emb_npy):
    df = pd.read_csv(csv_path)
    rgb_paths = df['url'].tolist()
    tactile_paths = df['tactile'].tolist()
    captions = df['caption'].tolist()
    indices = df['index'].tolist()
    # tactile_emb 需和df顺序一一对应
    tac_emb = np.load(tactile_emb_npy)  # shape=(N, D1)
    assert len(tac_emb) == len(df), "embedding数量与csv行数不一致"
    return tac_emb, captions, rgb_paths, tactile_paths, indices

# ========= 2. CLIP编码 =========

def get_clip_model(device="cuda"):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def encode_captions(captions, model, device):
    with torch.no_grad():
        text_tokens = clip.tokenize(captions, truncate=True).to(device)
        text_emb = model.encode_text(text_tokens).cpu().numpy()
        text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    return text_emb

def encode_images(img_paths, preprocess, model, device, batch=32):
    outs = []
    for i in tqdm(range(0, len(img_paths), batch)):
        imgs = [preprocess(Image.open(p).convert("RGB")) for p in img_paths[i:i+batch]]
        imgs = torch.stack(imgs).to(device)
        with torch.no_grad():
            out = model.encode_image(imgs).cpu().numpy()
            out = out / np.linalg.norm(out, axis=1, keepdims=True)
        outs.append(out)
    return np.vstack(outs)

# ========= 3. Tactile → Text投影MLP =========

class Tac2TextMLP(nn.Module):
    def __init__(self, tac_dim, text_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tac_dim, tac_dim),
            nn.ReLU(),
            nn.Linear(tac_dim, text_dim)
        )
    def forward(self, x):
        return self.mlp(x)

# ========= 4. 训练Tactile→Text投影 =========

def train_tac2text(tac_emb, text_emb, tac_dim, text_dim, device, epochs=20, batch_size=128, lr=1e-3, save_path="tac2text_proj.pt"):
    model = Tac2TextMLP(tac_dim, text_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    tac_tensor = torch.from_numpy(tac_emb).float().to(device)
    text_tensor = torch.from_numpy(text_emb).float().to(device)

    for epoch in range(epochs):
        perm = torch.randperm(len(tac_tensor))
        tac_tensor_epoch = tac_tensor[perm]
        text_tensor_epoch = text_tensor[perm]
        total_loss = 0
        for i in range(0, len(tac_tensor_epoch), batch_size):
            x = tac_tensor_epoch[i:i+batch_size]
            y = text_tensor_epoch[i:i+batch_size]
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(tac_tensor)
        print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"✅ 投影MLP已保存：{save_path}")
    return model

# ========= 5. 检索评测 =========

def evaluate_retrieval(tac_emb, tac2text_model, rgb_emb, device, batch=128, topk=10):
    tac2text_model.eval()
    tac_emb_tensor = torch.from_numpy(tac_emb).float().to(device)
    all_tac_text_emb = []
    with torch.no_grad():
        for i in range(0, len(tac_emb_tensor), batch):
            x = tac_emb_tensor[i:i+batch]
            out = tac2text_model(x).cpu().numpy()
            out = out / np.linalg.norm(out, axis=1, keepdims=True)
            all_tac_text_emb.append(out)
    tac_text_emb = np.vstack(all_tac_text_emb)

    # 构建faiss库
    faiss_index = faiss.IndexFlatIP(rgb_emb.shape[1])
    faiss_index.add(rgb_emb.astype("float32"))

    # 检索
    D, I = faiss_index.search(tac_text_emb.astype("float32"), topk)

    # 评估指标
    aps = []
    top1_correct = 0
    recall5 = 0
    for q_idx, retrieved in enumerate(I):
        gt = q_idx
        if gt in retrieved[:1]:
            top1_correct += 1
        if gt in retrieved[:5]:
            recall5 += 1
        y_true = np.zeros_like(retrieved)
        y_true[retrieved == gt] = 1
        y_score = D[q_idx]
        try:
            ap = average_precision_score(y_true, y_score)
        except:
            ap = 0
        aps.append(ap)
    print(f"\n🔎 检索评测：")
    print(f"tactile→rgb检索 mAP: {np.mean(aps):.4f}")
    print(f"Top-1: {top1_correct/len(I):.4f}")
    print(f"Recall@5: {recall5/len(I):.4f}")
    return np.mean(aps), top1_correct/len(I), recall5/len(I)

# ========= 6. 主流程 =========

if __name__ == "__main__":
    # 配置路径
    csv_file = "data/ssvtp/new_train.csv"              # 包含url、tactile、caption、index
    tactile_npy = "embeddings/new/all_embeddings.npy"  # shape=(N, D1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载数据
    tac_emb, captions, rgb_paths, tactile_paths, indices = load_from_csv(csv_file, tactile_npy)
    print("数据加载完成：", tac_emb.shape, len(captions), len(rgb_paths))
    captions = [str(c) if isinstance(c, str) or not pd.isna(c) else "" for c in captions]

    # 2. 加载CLIP
    clip_model, preprocess = get_clip_model(device)
    print("CLIP模型加载完成")

    # 3. 生成text embedding
    print("正在编码文本...")
    text_emb = encode_captions(captions, clip_model, device)
    # 假设你的图片都在 base_img_dir 下
    base_img_dir = "data/ssvtp"

    # 拼接绝对路径
    rgb_paths = [os.path.join(base_img_dir, p) if not os.path.isabs(p) else p for p in rgb_paths]

    print("正在编码图片...")
    rgb_emb = encode_images(rgb_paths, preprocess, clip_model, device)

    # 4. 训练Tactile→Text投影
    print("训练Tactile→Text投影MLP...")
    tac_dim, text_dim = tac_emb.shape[1], text_emb.shape[1]
    tac2text_model = train_tac2text(
        tac_emb, text_emb, tac_dim, text_dim,
        device, epochs=20, batch_size=128, lr=1e-3,
        save_path="tac2text_proj.pt"
    )

    # 5. 检索评测
    evaluate_retrieval(tac_emb, tac2text_model, rgb_emb, device)
