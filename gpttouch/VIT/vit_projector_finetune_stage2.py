import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Optional, List, Dict
import copy

"""
vit_projector_finetune_stage2.py

Purpose
-------
Stage-2 fine-tuning script for polishing a previously trained ViT projector.
It loads a Stage-1 best checkpoint, resumes training with a lower learning
rate and a short cosine schedule, and saves a refined checkpoint for final use.

Core features
-------------
- Loads best Stage-1 checkpoint and its `config` to rebuild the model.
- EMA model maintained for stability; final saved checkpoint is EMA weights.
- Stage-2 defaults use a very low LR and fewer epochs to avoid catastrophic
  drift from the Stage-1 solution.
- Same loss (InfoNCE) and retrieval-based early stopping (R@1 on val set).

Usage
-----
- Provide aligned embeddings (recommended) or original embeddings and
  id-alignment files.
- Set `resume_path` to the Stage-1 best checkpoint and run the script; it will
  emit a new `_best.pt` checkpoint when validation improves.

"""

# =============================
# 1) Dataset with optional sensor IDs
# =============================
class TactileRGBPairDataset(Dataset):
    def __init__(self, tac_embs: np.ndarray, rgb_embs: np.ndarray, sensor_ids: Optional[np.ndarray] = None):
        assert tac_embs.shape == rgb_embs.shape, "tac_embs and rgb_embs must have same shape"
        self.tac = torch.from_numpy(tac_embs).float()
        self.rgb = torch.from_numpy(rgb_embs).float()
        self.sensor_ids = None if sensor_ids is None else torch.from_numpy(sensor_ids).long()

    def __len__(self):
        return self.tac.size(0)

    def __getitem__(self, idx):
        sid = -1 if self.sensor_ids is None else int(self.sensor_ids[idx].item())
        return self.tac[idx], self.rgb[idx], sid


# =============================
# 2) Multi-sensor BatchSampler (σ% from a randomly picked dataset) — NO REPLACEMENT per epoch
# =============================
class MultiSensorBatchSampler(Sampler[List[int]]):
    def __init__(self, sensor_ids: torch.Tensor, batch_size: int, sigma: float = 0.75):
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
        emb_dim: int = 1024,
        num_tokens: int = 16,
        dim: int = 1024,
        depth: int = 24,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_sensors: int = 0,
        sensor_token_len: int = 5,
        learnable_tau: bool = False,
        init_tau: float = 0.07,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.emb_dim = emb_dim
        self.sensor_token_len = sensor_token_len if num_sensors > 0 else 0
        self.num_sensors = num_sensors

        self.proj_in = nn.Linear(emb_dim, num_tokens * dim)
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
        z_t = F.normalize(z_tac, dim=-1)
        z_r = F.normalize(z_rgb, dim=-1)
        logits = (z_t @ z_r.t()) / tau
        labels = torch.arange(z_t.size(0), device=z_t.device)
        loss_t2v = F.cross_entropy(logits, labels)
        loss_v2t = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_t2v + loss_v2t)


# =============================
# 5) Metrics
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

    mAP = torch.mean(1.0 / true_ranks).item()
    return {'R1': r_at(1), 'R5': r_at(5), 'R10': r_at(10), 'MedR': true_ranks.median().item(), 'MeanR': true_ranks.mean().item(), 'mAP': mAP}

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


# =============================
# 6) Stage-2 Finetune
# =============================

def finetune_projector(
    tac_embs: np.ndarray,
    rgb_embs: np.ndarray,
    sensor_ids: Optional[np.ndarray] = None,
    *,
    resume_path: str = 'tac_projector_vit4_best.pt',
    save_path: str = 'tac_projector_vit5.pt',
    lr: float = 3e-6,
    epochs: int = 60,
    batch_size: int = 48,
    sigma: float = 0.75,
    val_ratio: float = 0.1,
    weight_decay: float = 0.05,
    early_stop_patience: int = 10,
    use_amp: bool = True,
):
    assert os.path.exists(resume_path), f"Checkpoint not found: {resume_path}"

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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- load config from checkpoint to build model ----
    ckpt = torch.load(resume_path, map_location=device)
    cfg = ckpt.get('config', {})

    model = ViTProjector(
        emb_dim=cfg.get('emb_dim', tac_embs.shape[1]),
        num_tokens=cfg.get('num_tokens', 16),
        dim=cfg.get('dim', 1024),
        depth=cfg.get('depth', 24),
        heads=cfg.get('heads', 16),
        num_sensors=cfg.get('num_sensors', int(train_ds.sensor_ids.max().item()) + 1 if train_ds.sensor_ids is not None else 0),
        sensor_token_len=cfg.get('sensor_token_len', 5),
        learnable_tau=cfg.get('learnable_tau', False),
        init_tau=cfg.get('init_tau', 0.07),
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])

    # ----------- EMA
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False

    def update_ema(student, teacher, m=0.9995):
        with torch.no_grad():
            for ps, pt in zip(student.parameters(), teacher.parameters()):
                pt.data.mul_(m).add_(ps.data, alpha=1 - m)
    # -------------------
    # ---- data loader ----
    if train_ds.sensor_ids is not None:
        sampler = MultiSensorBatchSampler(train_ds.sensor_ids, batch_size=batch_size, sigma=sigma)
        loader = DataLoader(train_ds, batch_sampler=sampler)
    else:
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ---- optimizer (param grouping) ----
    criterion = ContrastiveLoss()
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=min(epochs, 50))
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
                tau = model.get_tau()
                loss = criterion(z_tac, rgb, tau)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                update_ema(model, ema_model, m=0.9995)

            bs = tac.size(0)
            total_loss += loss.item() * bs
            total_count += bs

        scheduler.step()
        avg_loss = total_loss / max(1, total_count)

        # ---- validation ----
        val_metrics = evaluate_retrieval(model, val_ds, device, batch_size=min(1024, max(64, batch_size)))
        # 验证
        # val_metrics = evaluate_retrieval(ema_model, val_ds, device, batch_size=min(1024, max(64, batch_size)))



        print(
            f"[Stage-2] Epoch {epoch+1:03d}/{epochs} | loss={avg_loss:.6f} | lr={scheduler.get_last_lr()[0]:.2e} | tau={model.get_tau().item():.4f} | "
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
            torch.save({'state_dict': model.state_dict(), 'config': cfg}, best_path)
            print(f"* 保存新最佳模型: {best_path} (R@1_t2i={best_r1:.3f})")
        else:
            epochs_no_improve += 1
            if early_stop_patience and epochs_no_improve >= early_stop_patience:
                print(f"早停: 连续 {early_stop_patience} 个 epoch 验证 R@1 无提升。")
                break

    # Save final weights
    # torch.save({'state_dict': model.state_dict(), 'config': cfg}, save_path)
    # 保存最佳
    torch.save({'state_dict': ema_model.state_dict(), 'config': cfg}, best_path)
    print(f"阶段二微调完成，模型已保存到: {save_path}")

    return model


if __name__ == "__main__":

    # -------------------------
    # safe load aligned if exists
    # -------------------------
    aligned_tac = "embeddings/embeddings_tac/all_embeddings_aligned.npy"
    aligned_rgb = "embeddings/embeddings_rgb/all_embeddings_aligned.npy"
    aligned_sid = "embeddings/sensor_ids_aligned.npy"  # optional
    aligned_pairs = "aligned_pairs.npz"

    if os.path.exists(aligned_tac) and os.path.exists(aligned_rgb):
        print("[LOAD] Found aligned embeddings -> using them for finetune (no re-align).")
        tac = np.load(aligned_tac)
        rgb = np.load(aligned_rgb)
        # sensor_ids = np.load(aligned_sid) if os.path.exists(aligned_sid) else sensor_ids

        # quick checks
        assert tac.ndim == 2 and rgb.ndim == 2, "Embeddings must be 2D arrays"
        assert tac.shape[0] == rgb.shape[0], f"Aligned N mismatch: tac {tac.shape[0]} vs rgb {rgb.shape[0]}"
        assert tac.shape[1] == rgb.shape[1], f"Dim mismatch: tac D={tac.shape[1]} vs rgb D={rgb.shape[1]}"

        # optional: load kept indices for traceability
        if os.path.exists(aligned_pairs):
            d = np.load(aligned_pairs)
            kept_t_idx = d["kept_t_idx"];
            kept_r_idx = d["kept_r_idx"]
            print(f"[LOAD] loaded aligned_pairs.npz (kept N={len(kept_t_idx)})")

        # small cosine diag sanity
        t = tac.astype("float32")
        r = rgb.astype("float32")
        t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
        r = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)
        diag = (t * r).sum(axis=1)
        print(f"[CHECK] diag sim mean={diag.mean():.4f}, median={np.median(diag):.4f}, std={diag.std():.4f}")
    else:
        print("[LOAD] aligned files not found — falling back to original paths and (re)align if needed.")
        tac = np.load(aligned_tac)
        rgb = np.load(aligned_rgb)
        # (keep your existing behavior to optionally align)

    # Example data paths (modify to yours)
    # tac_path = "embeddings/embeddings_tac/all_embeddings.npy"
    # rgb_path = "embeddings/embeddings_rgb/all_embeddings.npy"
    sid_path = "embeddings/sensor_ids.npy"
    #
    # tac = np.load(tac_path)
    # rgb = np.load(rgb_path)
    try:
        sensor_ids = np.load(sid_path)
    except Exception:
        sensor_ids = None
        print("未找到 sensor_ids.npy，将在无传感器标注的模式下微调（不使用专用 batch 采样与 sensor tokens）。")

    # finetune_projector(
    #     tac_embs=tac,
    #     rgb_embs=rgb,
    #     sensor_ids=sensor_ids,
    #     resume_path='tac_projector_vit2_best.pt',   # ← 你的 Stage-1 最优权重
    #     save_path='tac_projector_vit3.pt',          # ← 阶段二微调输出
    #     lr=3e-6,
    #     epochs=100,
    #     batch_size=48,
    #     sigma=0.75,
    #     val_ratio=0.1,
    #     weight_decay=0.05,
    #     early_stop_patience=20,
    #     use_amp=True,
    # )
    finetune_projector(
        tac_embs=tac,
        rgb_embs=rgb,
        sensor_ids=sensor_ids,
        resume_path='tac_projector_vit5_best.pt',  # 载入上一阶段最佳
        save_path='tac_projector_vit5p.pt',  # 抛光输出（p=polish）
        lr=3e-7,  # 更小 LR
        epochs=30,  # 小步快跑
        batch_size=48,
        sigma=0.75,
        val_ratio=0.1,
        weight_decay=0.05,
        early_stop_patience=10,  # 更激进的早停
        use_amp=True,
    )

