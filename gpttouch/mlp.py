# 文件名：train_tactile_projector.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ========== 1. 定义MLP投影网络 ==========
class TactileProjector(nn.Module):
    def __init__(self, emb_dim=1024):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):
        return self.projector(x)


# ========== 2. 数据集类 ==========
class TactileRGBPairDataset(Dataset):
    def __init__(self, tac_embs, rgb_embs):
        self.tac_embs = torch.from_numpy(tac_embs).float()
        self.rgb_embs = torch.from_numpy(rgb_embs).float()

    def __len__(self):
        return len(self.tac_embs)

    def __getitem__(self, idx):
        return self.tac_embs[idx], self.rgb_embs[idx]


# ========== 3. 训练函数 ==========
def train_projector(tac_embs, rgb_embs, lr=1e-3, epochs=50, batch_size=64, save_path="tac_projector.pt"):
    dataset = TactileRGBPairDataset(tac_embs, rgb_embs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    projector = TactileProjector(emb_dim=tac_embs.shape[1]).to(device)
    optimizer = torch.optim.Adam(projector.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        projector.train()
        total_loss = 0
        for tac, rgb in loader:
            tac, rgb = tac.to(device), rgb.to(device)
            tac_proj = projector(tac)
            loss = loss_fn(tac_proj, rgb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * tac.size(0)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataset):.6f}")

    torch.save(projector.state_dict(), save_path)
    print(f"投影网络已保存至 {save_path}")
    return projector


# ========== 4. 主流程 ==========
if __name__ == "__main__":
    # 修改为你的embedding路径
    tac_embs = np.load("embeddings/embeddings_tac/all_embeddings.npy")  # shape [N, 1024]
    rgb_embs = np.load("embeddings/embeddings_rgb/all_embeddings.npy")  # shape [N, 1024]

    projector = train_projector(
        tac_embs, rgb_embs,
        lr=1e-3,
        epochs=50,
        batch_size=64,
        save_path="tac_projector.pt"
    )
