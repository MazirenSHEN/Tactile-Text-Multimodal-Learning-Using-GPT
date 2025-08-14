import os
# 放在所有其它 import 之前
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 可选：限制线程数量，减少并发问题
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# test_vit_projector_with_index.py
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

CONFIG = {
    # embeddings
    "tactile_candidates": ["embeddings/test/test_tac.npy"],
    "rgb_candidates":     ["embeddings/test/test_rgb.npy"],
    # index (可选，但强烈推荐)
    "tac_idx_candidates": ["embeddings/test/test_tac_idx.npy"],
    "rgb_idx_candidates": ["embeddings/test/test_rgb_idx.npy"],
    # labels/text (zero-shot 可选)
    "labels_candidates":  ["embeddings/test/test_labels.npy"],
    "text_embs_candidates": ["embeddings/test/text_embs.npy"],
    # sensor ids (可选)
    "sensor_ids_candidates": ["embeddings/test/test_sensor_ids.npy", "embeddings/sensor_ids.npy"],
    # checkpoint
    "ckpt_candidates": ["tac_projector_vit5_best.pt"],
    "batch": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "zs_logit_scale": 20.0,
}

# ---------------- Model (与训练一致) ----------------
class ViTProjector(nn.Module):
    def __init__(self, emb_dim=1024, num_tokens=16, dim=1024, depth=24, heads=16,
                 mlp_ratio=4.0, dropout=0.0, num_sensors=0, sensor_token_len=5,
                 learnable_tau=False, init_tau=0.07):
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
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
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
        if hasattr(self, "log_tau") and self.log_tau is not None:
            return torch.exp(self.log_tau)
        return self.tau

    @torch.no_grad()
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

# ---------------- Helpers ----------------
def pick_first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if isinstance(p, str) and os.path.exists(p):
            return p
    return None

def load_checkpoint(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('config', {})
    model = ViTProjector(
        emb_dim=cfg.get('emb_dim', 1024),
        num_tokens=cfg.get('num_tokens', 16),
        dim=cfg.get('dim', 1024),
        depth=cfg.get('depth', 24),
        heads=cfg.get('heads', 16),
        num_sensors=cfg.get('num_sensors', 0),
        sensor_token_len=cfg.get('sensor_token_len', 5),
        learnable_tau=cfg.get('learnable_tau', False),
        init_tau=cfg.get('init_tau', 0.07),
    ).to(device).eval()
    model.load_state_dict(ckpt['state_dict'], strict=True)
    return model, cfg

@torch.no_grad()
def gather_projected(model: nn.Module, tac: np.ndarray, batch: int, device: str, sensor_ids: Optional[np.ndarray]):
    model.eval()
    outs = []
    N = tac.shape[0]
    for i in range(0, N, batch):
        x = torch.from_numpy(tac[i:i+batch]).float().to(device)
        if sensor_ids is not None:
            sid = torch.from_numpy(sensor_ids[i:i+batch]).long().to(device)
        else:
            sid = torch.full((x.size(0),), -1, device=device, dtype=torch.long)
        z = model(x, sensor_ids=sid)
        outs.append(z.detach().cpu())
    return torch.cat(outs, dim=0)

def _build_pos_map(indices: np.ndarray) -> Dict[int, List[int]]:
    """index value -> list of row positions"""
    pos_map: Dict[int, List[int]] = {}
    for pos, idx in enumerate(indices.tolist()):
        pos_map.setdefault(int(idx), []).append(pos)
    return pos_map

def _metrics_from_ranks(all_min_ranks: np.ndarray) -> Dict[str, float]:
    ranks = all_min_ranks.astype(np.float32)  # 1-based
    res = {
        "R1":  float(np.mean(ranks <= 1)),
        "R5":  float(np.mean(ranks <= 5)),
        "R10": float(np.mean(ranks <= 10)),
        "MedR": float(np.median(ranks)),
        "MeanR": float(np.mean(ranks)),
    }
    # AP（多正样本）我们计算严格 AP：遍历排序列表上的所有正样本，做 precision@k 平均
    # 这里在主循环里已经算好了 AP，可一起返回；为了保持简洁，我们在调用处返回。
    return res

@torch.no_grad()
def eval_with_indices(z_t: torch.Tensor, rgb: np.ndarray,
                      tac_idx: np.ndarray, rgb_idx: np.ndarray,
                      direction: str = "t2i") -> Dict[str, float]:
    """
    direction = 't2i' or 'i2t'
    多正样本支持：同一个 index 可能对应多条样本。
    计算：最小rank、R@K、MedR/MeanR，以及严格 AP（每个query按所有正样本的平均精度）。
    """
    if direction == "i2t":
        # 交换角色，重用实现
        return eval_with_indices(torch.from_numpy(rgb).float(), z_t.cpu().numpy(), rgb_idx, tac_idx, "t2i")

    # t2i
    Z = F.normalize(z_t, dim=-1).cpu().numpy()
    R = rgb.astype(np.float32)
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)

    rgb_map = _build_pos_map(rgb_idx)  # idx -> [positions]
    Nq = Z.shape[0]
    D = Z.shape[1]
    sims = Z @ R.T  # [Nq, Nr]

    min_ranks = []
    ap_list = []
    valid = 0
    for i in range(Nq):
        idx = int(tac_idx[i])
        pos_list = rgb_map.get(idx, [])
        if not pos_list:  # 没有正样本，跳过
            continue
        valid += 1
        row = sims[i]  # [Nr]

        # 降序排序得到排名
        order = np.argsort(-row)  # indices of sorted sims
        # 最小 rank
        ranks_of_pos = np.array([np.where(order == p)[0][0] + 1 for p in pos_list], dtype=np.int64)  # 1-based
        min_ranks.append(np.min(ranks_of_pos))

        # 严格 AP：对排序列表中每个“正样本”的位置 k，取 Precision@k 的平均
        hits = 0
        precisions = []
        pos_set = set(pos_list)
        for k, col in enumerate(order, start=1):
            if col in pos_set:
                hits += 1
                precisions.append(hits / k)
                if hits == len(pos_list):  # 全部正样本都已召回，可提前结束
                    break
        ap = float(np.mean(precisions)) if precisions else 0.0
        ap_list.append(ap)

    if valid == 0:
        return {"R1": 0.0, "R5": 0.0, "R10": 0.0, "MedR": float("inf"), "MeanR": float("inf"), "mAP": 0.0, "N": 0}
    min_ranks = np.array(min_ranks, dtype=np.float32)
    res = _metrics_from_ranks(min_ranks)
    res["mAP"] = float(np.mean(ap_list))
    res["N"] = int(valid)
    return res

@torch.no_grad()
def eval_fallback_pairwise(z_t: torch.Tensor, rgb: np.ndarray) -> Dict[str, float]:
    """无 index 时的按行一一对应评估（单正样本）"""
    Z = F.normalize(z_t, dim=-1)
    R = torch.from_numpy(rgb).float()
    R = F.normalize(R, dim=-1)
    sim = (Z @ R.t()).cpu().numpy()  # [N, N]

    order = np.argsort(-sim, axis=1)
    inv = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    inv[rows, order] = np.arange(order.shape[1])[None, :]
    ranks = inv[np.arange(inv.shape[0]), np.arange(order.shape[0])] + 1  # 1-based

    res = _metrics_from_ranks(ranks.astype(np.float32))
    # 单正样本 mAP ≈ mean(1/rank)（近似）
    res["mAP"] = float(np.mean(1.0 / ranks))
    res["N"] = int(sim.shape[0])
    return res

@torch.no_grad()
def eval_zeroshot(model, tac, text_embs, labels, batch, device, sensor_ids, logit_scale: float = 20.0):
    z_t = gather_projected(model, tac, batch, device, sensor_ids)
    Z = F.normalize(z_t, dim=-1)
    T = F.normalize(torch.from_numpy(text_embs).float(), dim=-1)
    logits = logit_scale * (Z @ T.t())
    preds = logits.topk(k=5, dim=1).indices.cpu().numpy()
    y = labels.astype(np.int64)
    top1 = (preds[:, 0] == y).mean().item()
    top5 = np.any(preds == y[:, None], axis=1).mean().item()
    return {'ZS_top1': float(top1), 'ZS_recall5': float(top5)}

def main():
    cfg = CONFIG
    # 选路径
    pick = lambda cands: next((p for p in cands if os.path.exists(p)), None)

    tac_path = pick(cfg["tactile_candidates"])
    rgb_path = pick(cfg["rgb_candidates"])
    tac_idx_path = pick(cfg.get("tac_idx_candidates", []))
    rgb_idx_path = pick(cfg.get("rgb_idx_candidates", []))
    sid_path = pick(cfg.get("sensor_ids_candidates", []))
    labels_path = pick(cfg.get("labels_candidates", []))
    text_path = pick(cfg.get("text_embs_candidates", []))
    ckpt_path = pick(cfg["ckpt_candidates"])

    assert tac_path and rgb_path and ckpt_path, "缺少必需文件（tac/rgb/ckpt）"

    # 加载数据
    tac = np.load(tac_path).astype(np.float32)
    rgb = np.load(rgb_path).astype(np.float32)
    assert tac.shape[1] == rgb.shape[1], f"维度不匹配：tac {tac.shape} vs rgb {rgb.shape}"

    tac_idx = np.load(tac_idx_path) if tac_idx_path else None
    rgb_idx = np.load(rgb_idx_path) if rgb_idx_path else None
    if (tac_idx is None) ^ (rgb_idx is None):
        print("[warn] 只找到一侧的 index，评估将退回按行对齐。")

    sensor_ids = np.load(sid_path) if sid_path else None
    labels = np.load(labels_path) if labels_path else None
    text_embs = np.load(text_path) if text_path else None

    # 加载模型并投影 tactile
    device = cfg["device"]
    model, mcfg = load_checkpoint(ckpt_path, device)
    z_t = gather_projected(model, tac, cfg["batch"], device, sensor_ids)

    # 评估
    if tac_idx is not None and rgb_idx is not None:
        # 基于 index 的“多正样本”评估
        t2i = eval_with_indices(z_t, rgb, tac_idx, rgb_idx, "t2i")
        i2t = eval_with_indices(z_t, rgb, tac_idx, rgb_idx, "i2t")
    else:
        # 回退：按行对齐（单正样本）
        t2i = eval_fallback_pairwise(z_t, rgb)
        i2t = eval_fallback_pairwise(torch.from_numpy(rgb).float(), z_t.cpu().numpy())

    # 打印
    print("\n== Paths ==")
    print("tactile:", tac_path)
    print("rgb    :", rgb_path)
    if tac_idx_path and rgb_idx_path:
        print("tac_idx:", tac_idx_path)
        print("rgb_idx:", rgb_idx_path)
    print("ckpt   :", ckpt_path)
    if sensor_ids is not None: print("sensor:", sid_path)
    if labels is not None and text_embs is not None:
        print("labels :", labels_path)
        print("text   :", text_path)

    print("\n== Retrieval (tactile→rgb) ==")
    print(f"R@1 {t2i['R1']:.3f} | R@5 {t2i['R5']:.3f} | R@10 {t2i['R10']:.3f} | mAP {t2i['mAP']:.3f} | "
          f"MedR {t2i['MedR']:.1f} | MeanR {t2i['MeanR']:.1f} | N {t2i['N']}")
    print("== Retrieval (rgb→tactile) ==")
    print(f"R@1 {i2t['R1']:.3f} | R@5 {i2t['R5']:.3f} | R@10 {i2t['R10']:.3f} | mAP {i2t['mAP']:.3f} | "
          f"MedR {i2t['MedR']:.1f} | MeanR {i2t['MeanR']:.1f} | N {i2t['N']}")

    results = {"t2i": t2i, "i2t": i2t}

    # Zero-shot（可选）
    if labels is not None and text_embs is not None and text_embs.shape[1] == tac.shape[1]:
        zs = eval_zeroshot(model, tac, text_embs, labels, cfg["batch"], device, sensor_ids, cfg["zs_logit_scale"])
        print("\n== Zero-shot classification ==")
        print(f"Top-1: {zs['ZS_top1']:.4f} | Recall@5: {zs['ZS_recall5']:.4f}")
        results["zeroshot"] = zs
    elif labels is not None or text_embs is not None:
        print("\n[warn] 跳过 zero-shot：text_embs 维度与视觉不匹配或缺少其一。")

    # 存结果
    out_json = os.path.splitext(ckpt_path)[0] + "_test_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {out_json}")

if __name__ == "__main__":
    main()
