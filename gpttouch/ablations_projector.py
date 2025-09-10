# ablations_projector_semantic_debug.py
# Caption-based positives with sentence-transformers semantic matching + diagnostics
import os
import argparse
import json
from typing import List, Dict, Optional, Tuple, Any

import re
import csv
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

from ablation_utils import compute_recall_at_k, compute_map, parse_tac_id_from_path
from TactileQASystem_integrated import TouchQAModel

# Optional semantic encoder (sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# ---------------------------
# Helpers
# ---------------------------
def build_paths(tactile_img_root: str, tac_rel_path: str) -> Tuple[str, Optional[str], Optional[int]]:
    tac_rel_path = (tac_rel_path or "").replace("\\", "/")
    tac_path = os.path.join(tactile_img_root, tac_rel_path)
    tid = parse_tac_id_from_path(tac_rel_path)
    rgb_path = None
    if tid is not None:
        rgb_path = os.path.join(tactile_img_root, f"images_rgb/image_{tid}_rgb.jpg")
    return tac_path, rgb_path, tid

def map_indices_to_tids(indices: List[int]) -> List[int]:
    return [int(i) for i in indices]

def norm_cap(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[\W_]+", " ", s)
    s = " ".join(s.split())
    return s

def _stats_from_ranks(ranks: List[int], ntotal_fallback: int = 1000) -> Dict[str, float]:
    if len(ranks) == 0:
        return {
            "R@1": 0.0, "R@5": 0.0, "R@10": 0.0,
            "mAP": 0.0, "MedR": float("inf"), "MeanR": float("inf"),
            "not_found_rate": 0.0, "N": 0,
        }
    out = {
        "R@1":  compute_recall_at_k(ranks, 1),
        "R@5":  compute_recall_at_k(ranks, 5),
        "R@10": compute_recall_at_k(ranks, 10),
        "mAP":  compute_map(ranks),
        "N":    len(ranks),
        "not_found_rate": float(sum(1 for r in ranks if r == 0) / len(ranks)),
    }
    max_seen = max([r for r in ranks if r > 0] + [ntotal_fallback])
    repl = [r if r > 0 else (max_seen + 1) for r in ranks]
    out["MedR"] = float(np.median(repl))
    out["MeanR"] = float(np.mean(repl))
    return out

def _bootstrap_ci(ranks: List[int], fn, n_boot: int = 2000, alpha: float = 0.05, seed: int = 1234):
    if len(ranks) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    ranks = np.asarray(ranks)
    vals = []
    for _ in range(n_boot):
        sample = ranks[rng.integers(0, len(ranks), size=len(ranks))]
        vals.append(fn(sample.tolist()))
    vals = np.asarray(vals, dtype=np.float64)
    return (float(vals.mean()),
            float(np.quantile(vals, alpha/2)),
            float(np.quantile(vals, 1-alpha/2)))

# ---------------------------
# Semantic matcher
# ---------------------------
class SemanticMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers not available; pip install sentence-transformers to use semantic matching")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    @staticmethod
    def cosine_sim(query_vec: np.ndarray, bank: np.ndarray) -> np.ndarray:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        sims = (query_vec @ bank.T).reshape(-1)
        return sims

# ---------------------------
# Caption-based positives with source detection
# ---------------------------

def get_test_caption_from_record(rec: Dict[str, Any]) -> str:
    for col in ("caption", "label", "text", "material"):
        val = rec.get(col)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""

def build_rgb_caption_map_and_list(idx2caption: Dict[str, str], rgb_ids: List[int]):
    cmap = {}
    caps_list = []
    for rid in rgb_ids:
        c = idx2caption.get(str(rid), "") if idx2caption else ""
        nc = norm_cap(c)
        cmap[int(rid)] = nc
        caps_list.append(nc)
    return cmap, caps_list

def token_overlap(a: str, b: str) -> float:
    sa = set(a.split()); sb = set(b.split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def find_positives_with_source(test_caption: str,
                                rgb_ids: List[int],
                                rgb_caps_list: List[str],
                                rgb_caption_map: Dict[int,str],
                                rgb_emb_bank: Optional[np.ndarray],
                                sem_model: Optional[SemanticMatcher],
                                semantic_thresh: float,
                                approx_thresh: float = 0.4,
                                semantic_topk: int = 10) -> Tuple[set, str]:
    """
    改进版：
      - 先 strict（规范化相等）
      - 若 strict 为空且启用了 sem_model：计算相似度，取 top-k（semantic_topk）
          - 从 top-k 中挑出 sim >= semantic_thresh 的作为 semantic_hits
          - 同时也可把 top-k 直接并入（可提高召回）；这里我们取 semantic_hits 且同时返回 topk_count 便于诊断
      - 最后 fallback token-overlap approx
    返回 (positives_set_of_rgb_ids, source) 其中 source ∈ {'strict','semantic','semantic_topk','approx','none'}
    """
    if not test_caption:
        return set(), 'none'
    key = norm_cap(test_caption)
    # 1) strict
    strict = set([rid for rid, ck in rgb_caption_map.items() if ck and ck == key])
    if strict:
        return strict, 'strict'

    # 2) semantic (top-k + threshold)
    if sem_model is not None and rgb_emb_bank is not None and len(rgb_emb_bank) > 0:
        try:
            q_emb = sem_model.encode([key])[0]  # (d,)
            sims = sem_model.cosine_sim(q_emb, rgb_emb_bank)  # (N,)
            # top-k indices
            topk = min(int(semantic_topk), len(sims))
            topk_idxs = np.argsort(-sims)[:topk]
            # semantic hits = those in top-k with sim >= semantic_thresh
            semantic_hits = [idx for idx in topk_idxs if float(sims[idx]) >= float(semantic_thresh)]
            semantic_set = set(int(rgb_ids[idx]) for idx in semantic_hits)
            if semantic_set:
                return semantic_set, 'semantic'
            # 若严格想提高召回：可以把 top-k 全部并入（作为 semantic_topk）
            # semantic_topk_set = set(int(rgb_ids[idx]) for idx in topk_idxs)
            # if semantic_topk_set:
            #     return semantic_topk_set, 'semantic_topk'
        except Exception:
            pass

    # 3) approx token-overlap fallback
    out = set()
    for rid, ck in rgb_caption_map.items():
        if not ck: continue
        if token_overlap(key, ck) >= approx_thresh:
            out.add(rid)
    if out:
        return out, 'approx'
    return set(), 'none'


# ---------------------------
# Core evaluation (returns diagnostics list as 4th element)
# ---------------------------
def evaluate_variant(
    qa_model: TouchQAModel,
    test_df: pd.DataFrame,
    tactile_img_root: str,
    faiss_topk: int = 1000,
    query_from: str = "tacdb",
    eval_mode: str = "bruteforce",
    t2t_mode: str = "caption",
    batch_size: int = 256,
    use_semantic: bool = False,
    semantic_model_name: str = "all-MiniLM-L6-v2",
    semantic_thresh: float = 0.65,
):
    rows: List[Dict[str, Any]] = []
    diag_semantic_only: List[Dict[str,Any]] = []

    # ---------- get arrays aligned with TouchQASystem ----------
    tac_raw, tac_ids, rgb_emb, rgb_ids = qa_model.get_arrays_for_eval()

    # captions
    caps_map = getattr(qa_model, "idx2caption", {}) or {}
    tac_caps = [caps_map.get(str(i), "") for i in tac_ids]
    cap_keys = np.array([norm_cap(c) for c in tac_caps])

    # Precompute Z and R so both branches can use them
    if getattr(qa_model, "projector", None) is not None:
        Z = qa_model.project_all_tactile(batch=batch_size)
    else:
        Z = tac_raw.astype("float32").copy(); faiss.normalize_L2(Z)
    R = rgb_emb.astype("float32").copy(); faiss.normalize_L2(R)

    # build rgb caption map & list
    rgb_caption_map, rgb_caps_list = build_rgb_caption_map_and_list(caps_map, rgb_ids)

    # semantic model
    sem_model = None
    rgb_emb_bank = None
    if use_semantic:
        if not _HAS_ST:
            print("Warning: sentence-transformers not installed; falling back to token-overlap matching.")
            use_semantic = False
        else:
            try:
                sem_model = SemanticMatcher(semantic_model_name)
                rgb_emb_bank = sem_model.encode(rgb_caps_list, batch_size=128)
                print(f"Loaded Semantic model '{semantic_model_name}', rgb caption bank size={len(rgb_caps_list)}")
            except Exception as e:
                print(f"Warning: failed to init semantic model '{semantic_model_name}': {e}. Falling back to token-overlap.")
                sem_model = None
                rgb_emb_bank = None
                use_semantic = False

    # ---------- BRUTEFORCE hybrid ----------
    if eval_mode == "bruteforce":
        id2row_tac: Dict[int, int] = {int(i): r for r, i in enumerate(tac_ids)}
        id2pos_rgb: Dict[int, List[int]] = {}
        for j, gid in enumerate(rgb_ids):
            id2pos_rgb.setdefault(int(gid), []).append(j)

        sim_tr = Z @ R.T
        order_tr = np.argsort(-sim_tr, axis=1)
        sim_tt = Z @ Z.T
        if t2t_mode != "self":
            np.fill_diagonal(sim_tt, -np.inf)
        order_tt = np.argsort(-sim_tt, axis=1)

        cap2pos: Dict[str, List[int]] = {}
        for j, ck in enumerate(cap_keys):
            cap2pos.setdefault(ck, []).append(j)

        query_indices = []
        query_records: List[Dict[str, Any]] = []
        fallback_rows = []

        for _, rec in test_df.iterrows():
            tac_rel = rec.get("tactile") or rec.get("image_path") or rec.get("img")
            if tac_rel is None:
                fallback_rows.append((rec, None)); continue
            _, _, tid = build_paths(tactile_img_root, tac_rel)
            if tid is None or int(tid) not in id2row_tac:
                fallback_rows.append((rec, tid)); continue
            query_indices.append(id2row_tac[int(tid)])
            query_records.append(rec)

        print(f"bruteforce hybrid: db_size={len(tac_ids)}, test_rows={len(test_df)}, matched_in_db={len(query_indices)}, fallback_rows={len(fallback_rows)}")

        ranks_t2r, aps_t2r = [], []
        for i, rec in zip(query_indices, query_records):
            test_cap = get_test_caption_from_record(rec)
            positives_rgb, src = find_positives_with_source(test_cap, rgb_ids, rgb_caps_list, rgb_caption_map, rgb_emb_bank, sem_model if use_semantic else None, semantic_thresh)
            # record diag if semantic-only
            if src == 'semantic':
                diag_semantic_only.append({"tactile": rec.get("tactile"), "test_caption": test_cap, "match_source": src})

            if not positives_rgb:
                ranks_t2r.append(0); continue

            pos_positions = []
            for rid in positives_rgb:
                pos_positions.extend(id2pos_rgb.get(int(rid), []))
            if not pos_positions:
                ranks_t2r.append(0); continue

            cols = order_tr[i]
            pos_set = set(pos_positions)
            min_rank = min(np.where(cols[:, None] == np.array(pos_positions))[0]) + 1
            ranks_t2r.append(int(min_rank))
            hits, precs = 0, []
            for k, c in enumerate(cols, start=1):
                if c in pos_set:
                    hits += 1; precs.append(hits / k)
                    if hits == len(pos_positions): break
            aps_t2r.append(float(np.mean(precs)) if precs else 0.0)

        ranks_t2t, aps_t2t, valid_t2t = [], [], 0
        for i, rec in zip(query_indices, query_records):
            test_cap = get_test_caption_from_record(rec)
            if t2t_mode == "self":
                pos = [i]
            else:
                pos = [j for j in cap2pos.get(norm_cap(test_cap), []) if j != i]

            if len(pos) == 0:
                rows.append({"tid": int(tac_ids[i]), "caption": tac_caps[i], "t2t_note": "no positives", "rank_t2t": 0})
                continue

            valid_t2t += 1
            cols = order_tt[i]
            pos_set = set(pos)
            min_rank = min(np.where(cols[:, None] == np.array(pos))[0]) + 1
            ranks_t2t.append(int(min_rank))
            hits, precs = 0, []
            for k, c in enumerate(cols, start=1):
                if c in pos_set:
                    hits += 1; precs.append(hits / k)
                    if hits == len(pos): break
            aps_t2t.append(float(np.mean(precs)) if precs else 0.0)

        topK = max(100, int(faiss_topk or 100))
        for rec, maybe_tid in fallback_rows:
            tac_rel = rec.get("tactile") or rec.get("image_path") or rec.get("img")
            if tac_rel is None:
                ranks_t2r.append(0); ranks_t2t.append(0); continue
            tac_path, _, tid = build_paths(tactile_img_root, tac_rel)

            try:
                if maybe_tid is not None and int(maybe_tid) in id2row_tac:
                    r = id2row_tac[int(maybe_tid)]
                    q_raw = qa_model.tac_raw[r].astype("float32")
                    q_l2  = qa_model.tac_emb[r].astype("float32")
                else:
                    if not os.path.exists(tac_path):
                        ranks_t2r.append(0); ranks_t2t.append(0); continue
                    q_raw = qa_model.extract_raw_embedding(tac_path)
                    q_l2 = q_raw.copy().astype("float32"); faiss.normalize_L2(q_l2.reshape(1,-1))
            except Exception as e:
                print(f"Warning: failed to extract embedding for {tac_rel}: {e}")
                ranks_t2r.append(0); ranks_t2t.append(0); continue

            test_cap = get_test_caption_from_record(rec)
            positives_rgb, src = find_positives_with_source(test_cap, rgb_ids, rgb_caps_list, rgb_caption_map, rgb_emb_bank, sem_model if use_semantic else None, semantic_thresh)
            if src == 'semantic':
                diag_semantic_only.append({"tactile": tac_rel, "test_caption": test_cap, "match_source": src})

            # t2r
            if getattr(qa_model, "projector", None) is not None and getattr(qa_model, "rgb_index", None) is not None:
                q_proj = qa_model.apply_projector_to_vector(q_raw)
                D, I = qa_model.rgb_index.search(q_proj.reshape(1,-1).astype("float32"), topK)
                tids_res = map_indices_to_tids(I[0].tolist())
                rank = 0
                for rnk, got in enumerate(tids_res, start=1):
                    if int(got) in positives_rgb:
                        rank = rnk; break
                ranks_t2r.append(rank)
            else:
                q_l2_row = q_l2.reshape(1,-1)
                sims = (q_l2_row @ R.T).reshape(-1)
                order = np.argsort(-sims)
                rank = 0
                if positives_rgb:
                    pospos = []
                    for rid in positives_rgb:
                        pospos.extend(id2pos_rgb.get(int(rid), []))
                    pospos_set = set(pospos)
                    for rnk, idx in enumerate(order, start=1):
                        if int(idx) in pospos_set:
                            rank = rnk; break
                ranks_t2r.append(rank)

            # t2t
            rank_tt = 0
            if getattr(qa_model, "projector", None) is not None and getattr(qa_model, "tac_proj_index", None) is not None:
                q_proj = qa_model.apply_projector_to_vector(q_raw)
                D, I = qa_model.tac_proj_index.search(q_proj.reshape(1,-1).astype("float32"), topK + (0 if t2t_mode=="self" else 1))
                tids_res = map_indices_to_tids(I[0].tolist())
            else:
                q_l2_row = q_l2.reshape(1,-1)
                sims = (q_l2_row @ Z.T).reshape(-1)
                order = np.argsort(-sims)
                tids_res = [int(tac_ids[idx]) for idx in order]

            if t2t_mode != "self":
                if maybe_tid is not None:
                    tids_res = [x for x in tids_res if int(x) != int(maybe_tid)]

            pos_tid_set = set()
            if test_cap:
                ck = norm_cap(test_cap)
                pos_indices = cap2pos.get(ck, [])
                pos_tid_set = set(int(tac_ids[idx]) for idx in pos_indices)

            for rnk, got in enumerate(tids_res, start=1):
                if int(got) in pos_tid_set:
                    rank_tt = rnk; break
            ranks_t2t.append(rank_tt)

        stats = {
            "R@1_t2r": compute_recall_at_k(ranks_t2r, 1),
            "R@5_t2r": compute_recall_at_k(ranks_t2r, 5),
            "R@10_t2r": compute_recall_at_k(ranks_t2r, 10),
            "mAP_t2r": float(np.mean(aps_t2r)) if aps_t2r else 0.0,
            "MedR_t2r": float(np.median([r if r > 0 else (len(rgb_ids)+1) for r in ranks_t2r])) if ranks_t2r else float("inf"),
            "MeanR_t2r": float(np.mean([r if r > 0 else (len(rgb_ids)+1) for r in ranks_t2r])) if ranks_t2r else float("inf"),
            "N": len(ranks_t2r),
            "not_found_rate_t2r": float(sum(1 for r in ranks_t2r if r == 0) / max(1, len(ranks_t2r))),
            "R@1_t2t": compute_recall_at_k(ranks_t2t, 1) if ranks_t2t else 0.0,
            "R@5_t2t": compute_recall_at_k(ranks_t2t, 5) if ranks_t2t else 0.0,
            "mAP_t2t": float(np.mean(aps_t2t)) if aps_t2t else 0.0,
            "MedR_t2t": float(np.median([r if r > 0 else (len(tac_ids)+1) for r in ranks_t2t])) if ranks_t2t else float("inf"),
            "MeanR_t2t": float(np.mean([r if r > 0 else (len(tac_ids)+1) for r in ranks_t2t])) if ranks_t2t else float("inf"),
            "not_found_rate_t2t": float(sum(1 for r in ranks_t2t if r == 0) / max(1, len(ranks_t2t))) if ranks_t2t else 0.0,
            "N_t2t_valid": int(sum(1 for r in ranks_t2t if r > 0)),
            "t2t_mode": t2t_mode,
        }
        stats_ci = {
            "R@1_t2r_CI": _bootstrap_ci(ranks_t2r, lambda rr: compute_recall_at_k(rr, 1)),
            "R@5_t2r_CI": _bootstrap_ci(ranks_t2r, lambda rr: compute_recall_at_k(rr, 5)),
            "mAP_t2r_CI": _bootstrap_ci(ranks_t2r, compute_map),
        }
        return rows, stats, stats_ci, diag_semantic_only

    # ---------- FAISS mode ----------
    ranks_t2r, ranks_t2t = [], []
    topK = max(100, int(faiss_topk or 100))

    id2row_tac = {int(i): r for r, i in enumerate(tac_ids)}
    cap2pos: Dict[str, List[int]] = {}
    for j, ck in enumerate(cap_keys):
        cap2pos.setdefault(ck, []).append(j)

    for _, rec in tqdm(test_df.iterrows(), total=len(test_df)):
        tac_rel = rec.get("tactile") or rec.get("image_path") or rec.get("img")
        if tac_rel is None:
            ranks_t2r.append(0); ranks_t2t.append(0); continue
        tac_path, _, tid = build_paths(tactile_img_root, tac_rel)

        if query_from == "tacdb" and tid in id2row_tac:
            r = id2row_tac[tid]
            q_raw = qa_model.tac_raw[r].astype("float32")
            q_l2  = qa_model.tac_emb[r].astype("float32")
        else:
            if not os.path.exists(tac_path):
                ranks_t2r.append(0); ranks_t2t.append(0); continue
            q_raw = qa_model.extract_raw_embedding(tac_path)
            q_l2 = q_raw.copy().astype("float32"); faiss.normalize_L2(q_l2.reshape(1,-1))

        test_cap = get_test_caption_from_record(rec)
        positives_rgb, src = find_positives_with_source(test_cap, rgb_ids, rgb_caps_list, rgb_caption_map, rgb_emb_bank, sem_model if use_semantic else None, semantic_thresh)
        if src == 'semantic':
            diag_semantic_only.append({"tactile": tac_rel, "test_caption": test_cap, "match_source": src})

        # t2r
        if getattr(qa_model, "projector", None) is not None:
            q_proj = qa_model.apply_projector_to_vector(q_raw)
            D, I = qa_model.rgb_index.search(q_proj.reshape(1,-1).astype("float32"), topK)
            tids = map_indices_to_tids(I[0].tolist())
            rank = 0
            for rnk, got in enumerate(tids, start=1):
                if int(got) in positives_rgb:
                    rank = rnk; break
            ranks_t2r.append(rank)
        else:
            q_l2_row = q_l2.reshape(1,-1)
            sims = (q_l2_row @ R.T).reshape(-1)
            order = np.argsort(-sims)
            id2pos_rgb: Dict[int, List[int]] = {}
            for j, gid in enumerate(rgb_ids):
                id2pos_rgb.setdefault(int(gid), []).append(j)
            pos_positions = []
            for rid in positives_rgb:
                pos_positions.extend(id2pos_rgb.get(int(rid), []))
            rank = 0
            if pos_positions:
                pos_set = set(pos_positions)
                for rnk, idx in enumerate(order, start=1):
                    if idx in pos_set:
                        rank = rnk; break
            ranks_t2r.append(rank)

        # t2t
        rank_tt = 0
        if getattr(qa_model, "projector", None) is not None and getattr(qa_model, "tac_proj_index", None) is not None:
            q_proj = qa_model.apply_projector_to_vector(q_raw)
            D, I = qa_model.tac_proj_index.search(q_proj.reshape(1,-1).astype("float32"), topK + (0 if t2t_mode=="self" else 1))
            tids_res = map_indices_to_tids(I[0].tolist())
        else:
            q_l2_row = q_l2.reshape(1,-1)
            sims = (q_l2_row @ Z.T).reshape(-1)
            order = np.argsort(-sims)
            tids_res = [int(tac_ids[idx]) for idx in order]

        if t2t_mode != "self":
            if tid is not None:
                tids_res = [x for x in tids_res if int(x) != int(tid)]

        pos_tid_set = set()
        if test_cap:
            ck = norm_cap(test_cap)
            pos_indices = cap2pos.get(ck, [])
            pos_tid_set = set(int(tac_ids[idx]) for idx in pos_indices)

        for rnk, got in enumerate(tids_res, start=1):
            if int(got) in pos_tid_set:
                rank_tt = rnk; break
        ranks_t2t.append(rank_tt)

    stats = {}
    stats.update({f"{k}_t2r": v for k, v in _stats_from_ranks(ranks_t2r, ntotal_fallback=len(rgb_ids)).items()})
    stats.update({f"{k}_t2t": v for k, v in _stats_from_ranks(ranks_t2t, ntotal_fallback=len(tac_ids)).items()})
    stats["t2t_mode"] = t2t_mode

    stats_ci = {
        "R@1_t2r_CI": _bootstrap_ci(ranks_t2r, lambda rr: compute_recall_at_k(rr, 1)),
        "R@5_t2r_CI": _bootstrap_ci(ranks_t2r, lambda rr: compute_recall_at_k(rr, 5)),
        "mAP_t2r_CI": _bootstrap_ci(ranks_t2r, compute_map),
    }
    return rows, stats, stats_ci, diag_semantic_only


def evaluate_t2r_only(
    qa_model: TouchQAModel,
    test_df: pd.DataFrame,
    tactile_img_root: str,
    faiss_topk: int = 1000,
    eval_mode: str = "bruteforce",    # "bruteforce" or "faiss"
    batch_size: int = 256,
    use_semantic: bool = True,
    semantic_model_name: str = "all-MiniLM-L6-v2",
    semantic_thresh: float = 0.75,
    semantic_topk: int = 10,
    semantic_allow_topk_fallback: bool = False,  # whether to accept topk even if below thresh
    return_diagnostics: bool = True,
):
    """
    Evaluate tactile -> rgb retrieval only (t2r) using caption semantic matching as positives.
    Returns: rows(list), stats(dict), stats_ci(dict), diag_semantic_only(list)
    """
    rows = []
    diag_semantic_only = []

    # load arrays aligned with model
    tac_raw, tac_ids, rgb_emb, rgb_ids = qa_model.get_arrays_for_eval()

    # captions map from model
    caps_map = getattr(qa_model, "idx2caption", {}) or {}
    # Normed caption keys for rgb db
    rgb_caption_map, rgb_caps_list = build_rgb_caption_map_and_list(caps_map, rgb_ids)

    # Precompute projected tac bank Z if projector available, else use tac_raw normalized
    if getattr(qa_model, "projector", None) is not None:
        Z = qa_model.project_all_tactile(batch=batch_size)  # [N_tac, D]
    else:
        Z = tac_raw.astype("float32").copy()
        faiss.normalize_L2(Z)

    # rgb bank R (normalized)
    R = rgb_emb.astype("float32").copy()
    faiss.normalize_L2(R)

    # Prepare semantic model if requested
    sem_model = None
    rgb_emb_bank = None
    if use_semantic:
        if not _HAS_ST:
            print("Warning: sentence-transformers not installed; falling back to token-overlap matching.")
            use_semantic = False
        else:
            try:
                sem_model = SemanticMatcher(semantic_model_name)
                rgb_emb_bank = sem_model.encode(rgb_caps_list, batch_size=128) if rgb_caps_list else None
                print(f"[semantic] loaded model {semantic_model_name}, rgb caption bank size={len(rgb_caps_list)}")
            except Exception as e:
                print(f"Warning: semantic model init failed: {e}. Falling back to token-overlap.")
                sem_model = None
                rgb_emb_bank = None
                use_semantic = False

    # Helper: get positives by semantic thresholding (only for rgb_ids list)
    def get_positives_for_caption(test_cap: str):
        key = norm_cap(test_cap)
        if not key:
            return set(), "none"

        # strict exact-match on normalized caption first
        strict = set([rid for rid, ck in rgb_caption_map.items() if ck and ck == key])
        if strict:
            return strict, "strict"

        # semantic
        if sem_model is not None and rgb_emb_bank is not None and len(rgb_emb_bank) > 0:
            try:
                q_emb = sem_model.encode([key])[0]   # (d,)
                sims = sem_model.cosine_sim(q_emb, rgb_emb_bank)  # shape (N,)
                # topk indices
                k = min(int(semantic_topk), len(sims))
                topk_idxs = np.argsort(-sims)[:k]
                # pick those >= threshold
                hits = [idx for idx in topk_idxs if float(sims[idx]) >= float(semantic_thresh)]
                if hits:
                    return set(int(rgb_ids[idx]) for idx in hits), "semantic"
                if semantic_allow_topk_fallback and topk_idxs.size > 0:
                    return set(int(rgb_ids[idx]) for idx in topk_idxs), "semantic_topk"
            except Exception:
                pass

        # fallback: token overlap heuristic (approx)
        out = set()
        for rid, ck in rgb_caption_map.items():
            if not ck: continue
            if token_overlap(key, ck) >= 0.4:
                out.add(rid)
        if out:
            return out, "approx"
        return set(), "none"

    # prepare id->positions mapping for RGB (for bruteforce)
    id2pos_rgb = {int(gid): [] for gid in rgb_ids}
    for j, gid in enumerate(rgb_ids):
        id2pos_rgb.setdefault(int(gid), []).append(j)

    # If using bruteforce, precompute similarities between Z and R to speed up (may be memory heavy)
    if eval_mode == "bruteforce":
        sim_tr = Z @ R.T   # [N_tac, N_rgb]
        order_tr = np.argsort(-sim_tr, axis=1)  # descending
    else:
        order_tr = None

    ranks = []
    aps = []
    diag_no_pos = []

    topK = max(100, int(faiss_topk or 100))

    # iterate test rows
    for _, rec in test_df.iterrows():
        tac_rel = rec.get("tactile") or rec.get("image_path") or rec.get("img")
        if tac_rel is None or str(tac_rel).strip() == "":
            ranks.append(0); aps.append(0.0); rows.append({"tactile": tac_rel, "note": "no tactile path", "rank_t2r": 0}); continue

        tac_path, _, tid = build_paths(tactile_img_root, tac_rel)

        # Try to obtain query embedding:
        q_raw = None; q_proj = None; q_l2 = None
        try:
            # if tid exists in tac_ids, use that row's precomputed Z/tac_emb to avoid re-extract
            if tid is not None and int(tid) in [int(x) for x in tac_ids]:
                row_idx = int(np.where(np.array(tac_ids)==int(tid))[0])
                if order_tr is not None:
                    cols = order_tr[row_idx]
                q_raw = qa_model.tac_raw[row_idx].astype("float32") if hasattr(qa_model, "tac_raw") else None
                q_l2 = qa_model.tac_emb[row_idx].astype("float32") if hasattr(qa_model, "tac_emb") else None
                if getattr(qa_model, "projector", None) is not None:
                    # use precomputed Z row
                    q_proj = Z[row_idx]
            else:
                # otherwise extract raw embedding from image file
                if not os.path.exists(tac_path):
                    ranks.append(0); aps.append(0.0); rows.append({"tactile": tac_rel, "note": "file missing", "rank_t2r": 0}); continue
                q_raw = qa_model.extract_raw_embedding(tac_path)
                q_l2 = q_raw.copy().astype("float32"); faiss.normalize_L2(q_l2.reshape(1, -1))
                if getattr(qa_model, "projector", None) is not None:
                    q_proj = qa_model.apply_projector_to_vector(q_raw)
        except Exception as e:
            print(f"Warning: failed to prepare query embedding for {tac_rel}: {e}")
            ranks.append(0); aps.append(0.0); rows.append({"tactile": tac_rel, "note": f"embed_fail:{e}", "rank_t2r": 0}); continue

        # build positives from caption semantics
        test_cap = get_test_caption_from_record(rec)
        positives_rgb, source = get_positives_for_caption(test_cap)
        if source == "semantic":
            diag_semantic_only.append({"tactile": tac_rel, "test_caption": test_cap, "match_source": source})
        if not positives_rgb:
            # keep record for diagnostics
            diag_no_pos.append({"tactile": tac_rel, "test_caption": test_cap})
            ranks.append(0); aps.append(0.0); rows.append({"tactile": tac_rel, "note": "no positives_from_caption", "rank_t2r": 0}); continue

        # DO the search (FAISS if requested & projector present, else brute)
        rank = 0
        prec_at_hits = []
        if eval_mode == "faiss" and getattr(qa_model, "projector", None) is not None and getattr(qa_model, "rgb_index", None) is not None:
            # use projector -> rgb_index search
            q_vec = q_proj.reshape(1, -1).astype("float32")
            D, I = qa_model.rgb_index.search(q_vec, topK)
            retrieved = map_indices_to_tids(I[0].tolist())
            # find first retrieved in positives_rgb
            for rnk, got in enumerate(retrieved, start=1):
                if int(got) in positives_rgb:
                    rank = rnk
                    break
            # compute AP
            hits = 0
            for k, got in enumerate(retrieved, start=1):
                if int(got) in positives_rgb:
                    hits += 1
                    prec_at_hits.append(hits / k)
                    if hits == len(positives_rgb):
                        break
            aps.append(float(np.mean(prec_at_hits)) if prec_at_hits else 0.0)
        else:
            # brute force similarity over R (use q_proj if exists else q_l2)
            if q_proj is not None:
                q_vec = q_proj.reshape(1, -1).astype("float32")
                sims = (q_vec @ R.T).reshape(-1)
            else:
                q_vec = q_l2.reshape(1, -1).astype("float32")
                sims = (q_vec @ R.T).reshape(-1)
            order = np.argsort(-sims)  # indices into R list
            # map positives rgb ids -> positions in R (id2pos_rgb)
            pos_positions = []
            for rid in positives_rgb:
                pos_positions.extend(id2pos_rgb.get(int(rid), []))
            # compute rank: first index in order that is in pos_positions
            pos_positions_set = set(pos_positions)
            mask_idx = np.where(np.isin(order, pos_positions))[0]
            rank = int(mask_idx[0]) + 1 if mask_idx.size > 0 else 0
            # compute AP
            hits = 0; precs = []
            for k, idx in enumerate(order, start=1):
                if int(idx) in pos_positions_set:
                    hits += 1; precs.append(hits / k)
                    if hits == len(pos_positions_set): break
            aps.append(float(np.mean(precs)) if precs else 0.0)

        ranks.append(int(rank))
        rows.append({"tactile": tac_rel, "caption": test_cap, "positives": list(positives_rgb), "match_source": source, "rank_t2r": int(rank)})

    # compute stats
    stats = {
        "R@1": compute_recall_at_k(ranks, 1),
        "R@5": compute_recall_at_k(ranks, 5),
        "R@10": compute_recall_at_k(ranks, 10),
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "MedR": float(np.median([r if r > 0 else (len(rgb_ids) + 1) for r in ranks])) if ranks else float("inf"),
        "MeanR": float(np.mean([r if r > 0 else (len(rgb_ids) + 1) for r in ranks])) if ranks else float("inf"),
        "N": len(ranks),
        "not_found_rate": float(sum(1 for r in ranks if r == 0) / max(1, len(ranks))),
    }

    stats_ci = {
        "R@1_CI": _bootstrap_ci(ranks, lambda rr: compute_recall_at_k(rr, 1)),
        "R@5_CI": _bootstrap_ci(ranks, lambda rr: compute_recall_at_k(rr, 5)),
        "mAP_CI": _bootstrap_ci(aps, lambda a: float(np.mean(a)) if len(a)>0 else 0.0) if aps else (0.0,0.0,0.0),
    }

    # diagnostics save
    if return_diagnostics and diag_semantic_only:
        # kept for caller to save to CSV
        pass

    return rows, stats, stats_ci, diag_semantic_only

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", type=str, default="data/ssvtp/test.csv")
    ap.add_argument("--tactile_emb_dir", type=str, default="embeddings/embeddings_tac")
    ap.add_argument("--rgb_emb_dir", type=str, default="embeddings/embeddings_rgb")
    ap.add_argument("--caption_csv", type=str, default="data/ssvtp/new_train.csv")
    ap.add_argument("--tactile_img_root", type=str, default="data/ssvtp")
    ap.add_argument("--variants_json", type=str, default="projector_variants.json")
    ap.add_argument("--out_dir", type=str, default="ablation_outputs/projector")
    ap.add_argument("--faiss_topk", type=int, default=1000)
    ap.add_argument("--query_from", type=str, default="tacdb", choices=["tacdb", "image"])
    ap.add_argument("--eval_mode", type=str, choices=["faiss", "bruteforce"], default="bruteforce")
    ap.add_argument("--t2t_mode", type=str, choices=["id", "caption", "self"], default="caption")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--use_semantic", action="store_true", help="enable sentence-transformers semantic caption matching")
    ap.add_argument("--semantic_model", type=str, default="all-MiniLM-L6-v2", help="sentence-transformers model name")
    ap.add_argument("--semantic_thresh", type=float, default=0.75, help="cosine threshold for semantic match (0-1)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.test_csv)

    if 'tactile' in df.columns: img_col = 'tactile'
    elif 'image_path' in df.columns: img_col = 'image_path'
    elif 'img' in df.columns: img_col = 'img'
    else: raise ValueError("test_csv must contain one of: 'tactile' | 'image_path' | 'img'")

    with open(args.variants_json, "r", encoding="utf-8") as f:
        variants = json.load(f)

    for name, v in variants.items():
        projector_path = v.get("projector_path")
        sensor_id = v.get("sensor_id", None)
        print(f"\n==== Evaluating variant: {name} (projector_path={projector_path}) ====")

        qa_model = TouchQAModel(
            tactile_emb_dir=args.tactile_emb_dir,
            rgb_emb_dir=args.rgb_emb_dir,
            projector_path=projector_path,
            caption_csv=args.caption_csv,
            tactile_img_dir=os.path.join(args.tactile_img_root, "images_tac"),
            rgb_img_dir=os.path.join(args.tactile_img_root, "images_rgb"),
            projector_sensor_id=sensor_id,
            normalize_before_projector=False,
        )

        test_df = df[df[img_col].astype(str).str.len() > 0].copy()
        # rows, stats, stats_ci, diag_sem = evaluate_variant(
        #     qa_model, test_df, args.tactile_img_root,
        #     faiss_topk=args.faiss_topk,
        #     query_from=args.query_from,
        #     eval_mode=args.eval_mode,
        #     t2t_mode=args.t2t_mode,
        #     batch_size=args.batch_size,
        #     use_semantic=args.use_semantic,
        #     semantic_model_name=args.semantic_model,
        #     semantic_thresh=args.semantic_thresh,
        # )
        rows, stats, stats_ci, diag_sem = evaluate_t2r_only(
            qa_model, test_df, args.tactile_img_root,
            faiss_topk=args.faiss_topk,
            eval_mode=args.eval_mode,  # "faiss" or "bruteforce"
            batch_size=args.batch_size,
            use_semantic=args.use_semantic,
            semantic_model_name=args.semantic_model,
            semantic_thresh=args.semantic_thresh,
            semantic_topk=10,
            semantic_allow_topk_fallback=False,
        )

        # save diagnostics
        if diag_sem:
            diag_csv = os.path.join(args.out_dir, f"{name}_semantic_only.csv")
            pd.DataFrame(diag_sem).to_csv(diag_csv, index=False, encoding="utf-8-sig")
            print(f"Saved semantic-only matches to: {diag_csv} (count={len(diag_sem)})")

        detail_csv = os.path.join(args.out_dir, f"{name}_details.csv")
        pd.DataFrame(rows).to_csv(detail_csv, index=False, encoding="utf-8-sig")
        summary_json = os.path.join(args.out_dir, f"{name}_summary.json")
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump({"stats": stats, "stats_ci": stats_ci}, f, ensure_ascii=False, indent=2)

        print("== Summary ==")
        print(json.dumps({"stats": stats, "stats_ci": stats_ci}, ensure_ascii=False, indent=2))
        print(f"Saved to: {detail_csv}\nSummary: {summary_json}")

if __name__ == "__main__":
    main()
