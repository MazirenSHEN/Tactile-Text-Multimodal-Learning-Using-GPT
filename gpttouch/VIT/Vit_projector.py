"""
Vit_projector.py

Purpose
-------
Training script for a UniTouch-style ViT projector that maps tactile (sensor)
embeddings to a target embedding space (typically RGB/ImageBind space).
The projector is a vector→vector ViT: it retokenizes a single embedding into a
short token sequence, runs a Transformer encoder, and reads out a CLS token
which is mapped back to the original embedding dimension.

Core features
-------------
- **Retokenization**: projects [B, D] → [B, T, C] to feed a Transformer.
- **Sensor tokens**: optional per-sensor learned tokens (and an unknown token)
  to model device-specific bias.
- **No-replacement multi-sensor batch sampler**: per-epoch balanced sampling
  with a controllable σ fraction from a main sensor.
- **ID alignment util**: `align_by_ids()` to build strict one-to-one pairs by
  intersecting tac/rgb id lists before training.
- **Training utilities**: param-grouping to avoid weight decay on norms/pos
  embeddings/sensor tokens/tau; AMP-enabled training; retrieval evaluation
  (R@1/R@5/R@10/MedR/MeanR/mAP) on a validation split.

Usage
-----
- Place precomputed embeddings and optional `*_indices.npy` and `sensor_ids.npy`
  under `embeddings/`.
- Run the training script directly; it will align ids (if provided) and then
  train a ViT projector saving the best checkpoint as `*_best.pt` and the final
  weights as specified.

Notes
-----
- Keep `emb_dim` consistent with your ImageBind or source embedding dimension.
- If you trained with sensor tokens, provide `sensor_ids.npy` during inference
  so the projector can use the matched token for the query.

"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Optional, List, Dict, Tuple

"""
UniTouch-style ViT Projector training script (with *number alignment* integrated). 
- **Align by IDs**: Before entering the Dataset, take the intersection of tac_indices and rgb_indices and re-arrange them to ensure that the positive samples used for training are truly one-to-one corresponding by the same ID.
- **Non-replacement multi-sensor sampling** (sigma = 0.75).
- **Optimizer grouping**: Do not apply weight decay to LayerNorm, bias, pos_emb, sensor_tokens, and tau.
- **AMP**, early stopping, and validation set retrieval metrics (R@K/MedR/MeanR/mAP).
- **Default hyperparameters for UniTouch**: dim = 1024, depth = 24, heads = 16, τ = 0.07 fixed, lr = 1e-5, epochs = 150. 
Usage: Run this file directly; modify the path of your npy file in the __main__ section.
"""

# =============================
# 1) Dataset with optional sensor IDs
# =============================
class TactileRGBPairDataset(Dataset):
    """
    tac_embs: np.ndarray [N, D]
    rgb_embs: np.ndarray [N, D] (frozen targets)
    sensor_ids: Optional[np.ndarray] [N] with integers in [0, K-1]
    """
    def __init__(self, tac_embs: np.ndarray, rgb_embs: np.ndarray, sensor_ids: Optional[np.ndarray] = None):
        assert tac_embs.shape == rgb_embs.shape, "tac_embs and rgb_embs must have same shape"
        self.tac = torch.from_numpy(tac_embs).float()
        self.rgb = torch.from_numpy(rgb_embs).float()
        if sensor_ids is None:
            self.sensor_ids = None
        else:
            self.sensor_ids = torch.from_numpy(sensor_ids).long()

    def __len__(self):
        return self.tac.size(0)

    def __getitem__(self, idx):
        if self.sensor_ids is None:
            return self.tac[idx], self.rgb[idx], -1  # -1 -> unknown sensor
        return self.tac[idx], self.rgb[idx], int(self.sensor_ids[idx].item())


# =============================
# 2) Multi-sensor BatchSampler (σ% from a randomly picked dataset) — NO REPLACEMENT per epoch
# =============================
class MultiSensorBatchSampler(Sampler[List[int]]):
    def __init__(self, sensor_ids: torch.Tensor, batch_size: int, sigma: float = 0.75):
        """
        sensor_ids: LongTensor [N], each in [0, K-1]
        sigma: fraction of a batch sampled from one chosen dataset each step
        """
        assert 0.0 <= sigma <= 1.0
        self.sensor_ids = sensor_ids
        self.batch_size = batch_size
        self.sigma = sigma

        self.N = sensor_ids.numel()
        self.indices_by_sensor: Dict[int, List[int]] = {}
        sensors = sensor_ids.unique(sorted=True).tolist()
        for s in sensors:
            idxs = (sensor_ids == s).nonzero(as_tuple=False).view(-1).tolist()
            self.indices_by_sensor[int(s)] = idxs

        # probabilities proportional to dataset sizes
        self.sensors = sorted(self.indices_by_sensor.keys())
        sizes = torch.tensor([len(self.indices_by_sensor[s]) for s in self.sensors], dtype=torch.float)
        probs = sizes / sizes.sum()
        self.probs = probs.tolist()

    def __len__(self):
        return math.ceil(self.N / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng()
        if len(self.sensors) == 0:
            perm = rng.permutation(self.N).tolist()
            for i in range(0, self.N, self.batch_size):
                yield perm[i:i + self.batch_size]
            return

        # Create per-sensor shuffled pools (no replacement within epoch)
        pools = {s: rng.permutation(self.indices_by_sensor[s]).tolist() for s in self.sensors}
        total = sum(len(v) for v in pools.values())
        num_batches = math.ceil(total / self.batch_size)

        for _ in range(num_batches):
            main_sensor = rng.choice(self.sensors, p=self.probs)
            n_main = int(round(self.batch_size * self.sigma))
            n_rest = self.batch_size - n_main

            main_take = []
            while pools[main_sensor] and len(main_take) < n_main:
                main_take.append(pools[main_sensor].pop())

            rest_take = []
            other_sensors = [s for s in self.sensors if s != main_sensor and len(pools[s]) > 0]
            rng.shuffle(other_sensors)
            while len(rest_take) < n_rest and (other_sensors or len(pools[main_sensor]) > 0):
                if other_sensors:
                    s = other_sensors[0]
                    rest_take.append(pools[s].pop())
                    if not pools[s]:
                        other_sensors.pop(0)
                elif pools[main_sensor]:
                    rest_take.append(pools[main_sensor].pop())
                else:
                    break

            batch = main_take + rest_take
            rng.shuffle(batch)
            if batch:
                yield batch


# =============================
# 3) ViT-style projector with (optional) sensor-specific tokens
# =============================
class ViTProjector(nn.Module):
    def __init__(
        self,
        emb_dim: int = 1024,   # input/output dimension (ImageBind=1024)
        num_tokens: int = 16,   # sequence length for retokenizing the vector
        dim: int = 1024,        # UniTouch uses C=1024
        depth: int = 24,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_sensors: int = 0,
        sensor_token_len: int = 5,
        learnable_tau: bool = False,
        init_tau: float = 0.07, # fixed tau per UniTouch
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.emb_dim = emb_dim
        self.sensor_token_len = sensor_token_len if num_sensors > 0 else 0
        self.num_sensors = num_sensors

        # retokenize: [B, D] -> [B, T, dim]
        self.proj_in = nn.Linear(emb_dim, num_tokens * dim)

        # tokens: cls + sensor(L) + T
        total_len = 1 + self.sensor_token_len + self.num_tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_emb = nn.Parameter(torch.randn(1, total_len, dim) * 0.02)

        if num_sensors > 0:
            self.sensor_tokens = nn.Parameter(torch.randn(num_sensors, self.sensor_token_len, dim) * 0.02)
            self.unk_sensor_tokens = nn.Parameter(torch.zeros(1, self.sensor_token_len, dim))
        else:
            self.sensor_tokens = None
            self.unk_sensor_tokens = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, emb_dim)

        if learnable_tau:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau)))
        else:
            self.register_buffer("tau", torch.tensor(init_tau))
            self.log_tau = None

        # init
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def get_tau(self):
        if self.log_tau is not None:
            return torch.exp(self.log_tau)
        return self.tau

    def forward(self, x: torch.Tensor, sensor_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.size(0)
        tokens = self.proj_in(x).view(B, self.num_tokens, self.dim)
        cls = self.cls_token.expand(B, -1, -1)

        if self.sensor_tokens is not None and sensor_ids is not None:
            sid = sensor_ids.clone()
            sid[sid < 0] = -1
            known_mask = sid >= 0
            out = torch.empty(B, self.sensor_token_len, self.dim, device=x.device, dtype=x.dtype)
            if known_mask.any():
                out[known_mask] = self.sensor_tokens[sid[known_mask]]
            if (~known_mask).any():
                out[~known_mask] = self.unk_sensor_tokens.expand((~known_mask).sum(), -1, -1)
            sensor_tok = out
        else:
            sensor_tok = torch.zeros(B, 0, self.dim, device=x.device, dtype=x.dtype)

        seq = torch.cat([cls, sensor_tok, tokens], dim=1)
        seq = seq + self.pos_emb[:, : seq.size(1), :]
        enc = self.encoder(seq)
        cls_out = self.norm(enc[:, 0])
        y = self.head(cls_out)
        return y


# =============================
# 4) Contrastive (InfoNCE) loss for alignment
# =============================
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_tac: torch.Tensor, z_rgb: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # cosine similarity -> [B, B]
        z_t = F.normalize(z_tac, dim=-1)
        z_r = F.normalize(z_rgb, dim=-1)
        logits = (z_t @ z_r.t()) / tau
        labels = torch.arange(z_t.size(0), device=z_t.device)
        loss_t2v = F.cross_entropy(logits, labels)
        loss_v2t = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_t2v + loss_v2t)


# =============================
# 5) Utilities: metrics & prototypes & id alignment
# =============================
@torch.no_grad()
def _gather_embeddings(model: nn.Module, dataset: Dataset, batch_size: int, device: str):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    z_t_list, z_r_list = [], []
    model.eval()
    for tac, rgb, sid in loader:
        tac = tac.to(device)
        rgb = rgb.to(device)
        sid = torch.as_tensor(sid, device=device) if not torch.is_tensor(sid) else sid.to(device)
        z_t = model(tac, sensor_ids=sid)
        z_t_list.append(z_t.detach().cpu())
        z_r_list.append(rgb.detach().cpu())
    z_t = torch.cat(z_t_list, dim=0)
    z_r = torch.cat(z_r_list, dim=0)
    return z_t, z_r

@torch.no_grad()
def _rank_stats(sim: torch.Tensor):
    N = sim.size(0)
    ranks = torch.argsort(sim, dim=1, descending=True)
    targets = torch.arange(N).view(-1, 1)
    pos = (ranks == targets).nonzero(as_tuple=False)
    _, rank_idx = torch.sort(pos[:,0])
    true_ranks = pos[rank_idx, 1].float() + 1

    def r_at(k):
        k = min(k, N)
        return (true_ranks <= k).float().mean().item()

    # simple mAP proxy for single-positive per query (1/rank)
    mAP = torch.mean(1.0 / true_ranks).item()

    return {
        'R1': r_at(1), 'R5': r_at(5), 'R10': r_at(10),
        'MedR': true_ranks.median().item(), 'MeanR': true_ranks.mean().item(), 'mAP': mAP,
    }

@torch.no_grad()
def evaluate_retrieval(model: nn.Module, val_ds: Dataset, device: str, batch_size: int = 512) -> dict:
    z_t, z_r = _gather_embeddings(model, val_ds, batch_size, device)
    z_t = F.normalize(z_t, dim=-1)
    z_r = F.normalize(z_r, dim=-1)
    sim = z_t @ z_r.t()
    stats_t = _rank_stats(sim)
    stats_i = _rank_stats(sim.t())
    return {
        'R1_t2i': stats_t['R1'], 'R5_t2i': stats_t['R5'], 'R10_t2i': stats_t['R10'],
        'MedR_t2i': stats_t['MedR'], 'MeanR_t2i': stats_t['MeanR'], 'mAP_t2i': stats_t['mAP'],
        'R1_i2t': stats_i['R1'], 'R5_i2t': stats_i['R5'], 'R10_i2t': stats_i['R10'],
        'MedR_i2t': stats_i['MedR'], 'MeanR_i2t': stats_i['MeanR'], 'mAP_i2t': stats_i['mAP'],
        'avg_sim': sim.diag().mean().item(), 'N': sim.size(0)
    }

@torch.no_grad()
def compute_sensor_prototypes(tac_embs: np.ndarray, sensor_ids: Optional[np.ndarray]):
    if sensor_ids is None:
        return None
    K = int(sensor_ids.max()) + 1
    D = tac_embs.shape[1]
    protos = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        idxs = np.where(sensor_ids == k)[0]
        if len(idxs) == 0:
            continue
        mu = tac_embs[idxs].mean(axis=0)
        # normalize prototype to unit length for cosine usage
        n = np.linalg.norm(mu) + 1e-8
        protos[k] = mu / n
    return protos

# --------- NEW: align by IDs (one-to-one by intersection) ---------

def align_by_ids(
    tac: np.ndarray,
    tac_ids: np.ndarray,
    rgb: np.ndarray,
    rgb_ids: np.ndarray,
    sensor_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """对齐触觉与图像嵌入：以编号交集建立一一对应。
    返回：tac_aln, rgb_aln, sensor_aln, kept_t_idx, kept_r_idx  （kept_* 是原数组中的索引）
    """
    assert tac.shape[1] == rgb.shape[1], "Embedding dim must match"
    tac_map = {int(i): idx for idx, i in enumerate(tac_ids.tolist())}
    keep = []
    for j, rid in enumerate(rgb_ids.tolist()):
        rid = int(rid)
        if rid in tac_map:
            keep.append((tac_map[rid], j))
    if not keep:
        raise ValueError("No overlapping ids between tac_ids and rgb_ids")
    keep.sort()  # 按 tactile 的顺序
    t_idx = np.array([a for a, _ in keep], dtype=int)
    r_idx = np.array([b for _, b in keep], dtype=int)
    tac_aln = tac[t_idx]
    rgb_aln = rgb[r_idx]
    sid_aln = sensor_ids[t_idx] if sensor_ids is not None else None
    return tac_aln, rgb_aln, sid_aln, t_idx, r_idx


# =============================
# 6) Training
# =============================

def train_projector(
    tac_embs: np.ndarray,
    rgb_embs: np.ndarray,
    sensor_ids: Optional[np.ndarray] = None,
    *,
    lr: float = 1e-5,            # UniTouch: 1e-5
    epochs: int = 150,           # UniTouch: 150
    batch_size: int = 48,        # UniTouch per-GPU batch
    save_path: str = "tac_projector_vit_unitouch.pt",
    sigma: float = 0.75,         # UniTouch: 0.75
    num_tokens: int = 16,
    dim: int = 1024,             # UniTouch C=1024
    depth: int = 24,             # UniTouch depth
    heads: int = 16,             # UniTouch heads
    sensor_token_len: int = 5,   # UniTouch L=5
    learnable_tau: bool = False, # UniTouch fixes tau
    init_tau: float = 0.07,      # UniTouch tau
    weight_decay: float = 0.05,
    val_ratio: float = 0.1,
    early_stop_patience: int = 10,
    use_amp: bool = True,
):
    # ---- split train/val ----
    N = tac_embs.shape[0]
    idx = np.arange(N)
    np.random.default_rng().shuffle(idx)
    n_val = max(1, int(N * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    tac_train, rgb_train = tac_embs[train_idx], rgb_embs[train_idx]
    tac_val, rgb_val = tac_embs[val_idx], rgb_embs[val_idx]
    sid_train = sensor_ids[train_idx] if sensor_ids is not None else None
    sid_val = sensor_ids[val_idx] if sensor_ids is not None else None

    train_ds = TactileRGBPairDataset(tac_train, rgb_train, sid_train)
    val_ds = TactileRGBPairDataset(tac_val, rgb_val, sid_val)

    if train_ds.sensor_ids is not None:
        sampler = MultiSensorBatchSampler(train_ds.sensor_ids, batch_size=batch_size, sigma=sigma)
        loader = DataLoader(train_ds, batch_sampler=sampler)
        num_sensors = int(train_ds.sensor_ids.max().item()) + 1
    else:
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        num_sensors = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTProjector(
        emb_dim=tac_embs.shape[1],
        num_tokens=num_tokens,
        dim=dim,
        depth=depth,
        heads=heads,
        num_sensors=num_sensors,
        sensor_token_len=sensor_token_len,
        learnable_tau=learnable_tau,
        init_tau=init_tau,
    ).to(device)

    criterion = ContrastiveLoss()

    # param grouping: no decay for norms/bias/pos/sensor/tau
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith('bias') or 'norm' in n.lower() or 'ln' in n.lower():
            no_decay.append(p)
        elif any(kw in n for kw in ['pos_emb', 'cls_token', 'sensor_tokens', 'unk_sensor_tokens', 'log_tau', 'tau']):
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.startswith('cuda')))

    best_r1 = -1.0
    best_path = save_path.replace('.pt', '_best.pt')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0

        for tac, rgb, sid in loader:
            tac = tac.to(device)
            rgb = rgb.to(device)
            sid = torch.as_tensor(sid, device=device) if not torch.is_tensor(sid) else sid.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith('cuda'))):
                z_tac = model(tac, sensor_ids=sid)
                tau = model.get_tau()  # keep tensor
                loss = criterion(z_tac, rgb, tau)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            bs = tac.size(0)
            total_loss += loss.item() * bs
            total_count += bs

        scheduler.step()
        avg_loss = total_loss / max(1, total_count)

        # ---- validation ----
        val_metrics = evaluate_retrieval(model, val_ds, device, batch_size=min(1024, max(64, batch_size)))
        print(
            f"Epoch {epoch+1:03d}/{epochs} | loss={avg_loss:.6f} | lr={scheduler.get_last_lr()[0]:.2e} | tau={model.get_tau().item():.4f} | "
            f"val R@1_t2i={val_metrics['R1_t2i']:.3f} R@5_t2i={val_metrics['R5_t2i']:.3f} R@10_t2i={val_metrics['R10_t2i']:.3f} | "
            f"mAP_t2i={val_metrics['mAP_t2i']:.3f} | "
            f"R@1_i2t={val_metrics['R1_i2t']:.3f} R@5_i2t={val_metrics['R5_i2t']:.3f} R@10_i2t={val_metrics['R10_i2t']:.3f} | "
            f"mAP_i2t={val_metrics['mAP_i2t']:.3f} | "
            f"MedR_t2i={val_metrics['MedR_t2i']:.1f} MedR_i2t={val_metrics['MedR_i2t']:.1f}"
        )

        # ---- early stopping & best ckpt ----
        if val_metrics['R1_t2i'] > best_r1:
            best_r1 = val_metrics['R1_t2i']
            epochs_no_improve = 0
            torch.save({'state_dict': model.state_dict(), 'config': {
                'emb_dim': tac_embs.shape[1], 'num_tokens': num_tokens, 'dim': dim, 'depth': depth,
                'heads': heads, 'num_sensors': num_sensors, 'sensor_token_len': sensor_token_len,
                'learnable_tau': learnable_tau, 'init_tau': init_tau,
            }}, best_path)
            print(f"* 保存新最佳模型到: {best_path} (R@1_t2i={best_r1:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"早停: 连续 {early_stop_patience} 个 epoch 验证 R@1 无提升。")
                break

    # Save weights + small training meta
    save_obj = {
        'state_dict': model.state_dict(),
        'config': {
            'emb_dim': tac_embs.shape[1],
            'num_tokens': num_tokens,
            'dim': dim,
            'depth': depth,
            'heads': heads,
            'num_sensors': num_sensors,
            'sensor_token_len': sensor_token_len,
            'learnable_tau': learnable_tau,
            'init_tau': init_tau,
        }
    }
    torch.save(save_obj, save_path)
    print(f"最终模型已保存到: {save_path}")

    # Save sensor prototypes for future nearest-neighbor sensor selection (optional)
    if sensor_ids is not None:
        protos = compute_sensor_prototypes(tac_embs, sensor_ids)
        if protos is not None:
            np.save(save_path.replace('.pt', '_sensor_prototypes.npy'), protos)
            print("已保存 sensor 原型: ", save_path.replace('.pt', '_sensor_prototypes.npy'))

    return model


if __name__ == "__main__":
    # ====== 你的数据路径（可按需修改）======
    tac_path = "embeddings/embeddings_tac/all_embeddings.npy"      # [N, D]
    rgb_path = "embeddings/embeddings_rgb/all_embeddings.npy"      # [N, D]
    tac_idx_path = "embeddings/embeddings_tac/all_indices.npy"     # [N] 触觉编号（可选，但强烈建议）
    rgb_idx_path = "embeddings/embeddings_rgb/all_indices.npy"     # [N] 图像编号（可选，但强烈建议）
    sid_path = "embeddings/sensor_ids.npy"                          # [N] 传感器ID（可选）

    tac = np.load(tac_path)
    rgb = np.load(rgb_path)

    tac_indices = np.load(tac_idx_path) if os.path.exists(tac_idx_path) else None
    rgb_indices = np.load(rgb_idx_path) if os.path.exists(rgb_idx_path) else None

    try:
        sensor_ids = np.load(sid_path)
    except Exception:
        sensor_ids = None
        print("未找到 sensor_ids.npy，将在无传感器标注的模式下训练（不使用 batch 采样策略和传感器 tokens）。")

    # ====== NEW: 训练前的编号对齐 ======
    if tac_indices is not None and rgb_indices is not None:
        print("\n[对齐] 检测到编号文件，开始按交集对齐...")
        tac_aln, rgb_aln, sid_aln, kept_t_idx, kept_r_idx = align_by_ids(
            tac, tac_indices, rgb, rgb_indices, sensor_ids
        )
        print(f"[对齐] tac 原始: {tac.shape[0]}，rgb 原始: {rgb.shape[0]} -> 对齐后 N: {tac_aln.shape[0]}")
        tac = tac_aln
        rgb = rgb_aln
        sensor_ids = sid_aln
        # 可选：把对齐后的索引保存下来，便于复现实验
        np.save("aligned_kept_tac_idx.npy", kept_t_idx)
        np.save("aligned_kept_rgb_idx.npy", kept_r_idx)
        # 对齐完成后（紧接着回写 tac/rgb/sensor_ids 之后）
        np.save("embeddings/embeddings_tac/all_embeddings_aligned.npy", tac)
        np.save("embeddings/embeddings_rgb/all_embeddings_aligned.npy", rgb)
        if sensor_ids is not None:
            np.save("embeddings/sensor_ids_aligned.npy", sensor_ids)

        # 也可以把对齐后的“成对映射”写成一个 npz，方便评测脚本直接复用
        np.savez("aligned_pairs.npz",
                 kept_t_idx=kept_t_idx,
                 kept_r_idx=kept_r_idx)

        print("[对齐] 已保存 kept 索引到 aligned_kept_*.npy")
    else:
        print("[对齐] 未提供编号文件，默认假定 tac 与 rgb 已按相同顺序一一对应！")

    # ====== 开始训练 ======
    train_projector(
        tac, rgb, sensor_ids,
        lr=1e-5,
        epochs=150,
        batch_size=48,
        save_path="vitmodel/tac_projector_vit2.pt",
        sigma=0.75,
        num_tokens=16,
        dim=1024,
        depth=24,
        heads=16,
        sensor_token_len=5,
        learnable_tau=False,
        init_tau=0.07,
        weight_decay=0.05,
        val_ratio=0.1,
        early_stop_patience=20,
        use_amp=True,
    )
