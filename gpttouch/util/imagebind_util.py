# build_train_embeddings_from_csv.py
"""
Build RGB & tactile embeddings from a CSV with columns: url, tactile, caption, index.

CSV requirements
----------------
- url:     RGB image path (local path; optionally supports http/https download if enabled)
- tactile: tactile image path (local path; optionally supports http/https download if enabled)
- caption: class/label text
- index:   sample id (will be saved and used to pair rgb/tactile rows)

Outputs
-------
- <out_rgb_dir>/all_embeddings.npy : float32 [N, D]
- <out_rgb_dir>/all_indices.npy    : indices (int64 if possible, else object)
- <out_tac_dir>/all_embeddings.npy : float32 [N, D]
- <out_tac_dir>/all_indices.npy    : indices (same as above)
- <out_root>/train_labels.npy      : per-sample class ids (int64)
- <out_root>/text_embs.npy         : class-level text embeddings (L2-normalized)
- <out_root>/manifest_rgb.csv      : mapping {index, resolved_rgb_path}
- <out_root>/manifest_tac.csv      : mapping {index, resolved_tactile_path}
- (optional) per-sample embeddings under per-sample dirs

Notes
-----
- Encoders: prefers ImageBind (imagebind_huge). Falls back to CLIP ViT-B/32 if ImageBind is unavailable.
- Embeddings are row-wise L2-normalized by default (recommended for FAISS IP/cosine).
- If a CSV path points to a .npy file (either a single [N,D] array or per-sample 1D arrays),
  the script will load instead of re-encoding images.
"""

import os
import sys
import csv
import pathlib
import shutil
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

# ---------------- Backend availability ----------------
HAS_IMAGEBIND = False
try:
    from imagebind.models.imagebind_model import imagebind_huge, ModalityType
    from imagebind.data import load_and_transform_vision_data
    HAS_IMAGEBIND = True
except Exception:
    HAS_IMAGEBIND = False

HAS_CLIP = False
try:
    import clip  # openai/CLIP
    HAS_CLIP = True
except Exception:
    HAS_CLIP = False

HAS_S2 = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_S2 = True
except Exception:
    HAS_S2 = False

# ---------------- CONFIG ----------------
CONFIG = {
    # CSV path with columns: url, tactile, caption, index
    "csv_path": "data/ssvtp/new_train.csv",

    # Candidate roots for resolving relative paths inside the CSV
    "candidate_roots": [
        ".",
        "data/ssvtp",
    ],

    # Optional: support http(s) URLs by downloading to a local cache
    "allow_http_download": False,
    "download_cache_dir": "_cache/downloads",  # used only if allow_http_download=True

    # Output root & subdirs
    "out_root": "embeddings/train",
    "out_rgb_dir": "embeddings/train/embeddings_rgb",
    "out_tac_dir": "embeddings/train/embeddings_tac",

    # Output filenames
    "out_all_emb": "all_embeddings.npy",
    "out_all_idx": "all_indices.npy",
    "out_labels": "train_labels.npy",
    "out_text_embs": "text_embs.npy",
    "out_manifest_rgb": "manifest_rgb.csv",
    "out_manifest_tac": "manifest_tac.csv",

    # Optional per-sample saves
    "save_per_sample_rgb": False,
    "save_per_sample_tac": False,
    "per_sample_rgb_dir": "embeddings/embeddings_rgb/per_image",
    "per_sample_tac_dir": "embeddings/embeddings_tac/per_image",

    # Device & batching
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_batch": 128,
    "text_batch": 64,

    # Encoder preference
    "prefer_image_encoder": "imagebind" if HAS_IMAGEBIND else ("clip" if HAS_CLIP else None),
    "prefer_text_encoder": "s2" if HAS_S2 else ("clip" if HAS_CLIP else None),

    # Normalize row-wise to unit length
    "l2_normalize": True,
}
# ------------------------------------------------------


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def ensure_out(root: str, name: str) -> str:
    if os.path.isabs(name):
        ensure_dir(os.path.dirname(name))
        return name
    out = os.path.join(root, name)
    ensure_dir(os.path.dirname(out))
    return out


def canonical_roots(cfg: dict) -> List[str]:
    roots = list(cfg.get("candidate_roots", []))
    csv_dir = os.path.dirname(cfg["csv_path"]) or "."
    roots = [str(pathlib.Path(r)) for r in ([".", csv_dir] + roots)]
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for r in roots:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq


def is_http_path(p: str) -> bool:
    p = p.lower()
    return p.startswith("http://") or p.startswith("https://")


def fetch_http_to_cache(url: str, cache_dir: str) -> Optional[str]:
    try:
        import hashlib
        import requests
        ensure_dir(cache_dir)
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
        local = os.path.join(cache_dir, f"{h}{ext}")
        if not os.path.exists(local):
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            with open(local, "wb") as f:
                f.write(r.content)
        return local
    except Exception as e:
        print(f"[warn] failed to fetch {url}: {e}")
        return None


def resolve_paths(items: List[str], roots: List[str], allow_http=False, cache_dir="_cache/downloads"):
    """Resolve possibly-relative paths against candidate roots; optionally fetch http(s).
    Returns: resolved(list[str or None]), missing(list[str])
    """
    resolved, missing = [], []
    for p in items:
        p = ("" if p is None else str(p).strip())
        if p == "":
            resolved.append(None)
            missing.append(p)
            continue
        # Remote URL
        if allow_http and is_http_path(p):
            local = fetch_http_to_cache(p, cache_dir)
            if local and os.path.exists(local):
                resolved.append(local)
                continue
            resolved.append(None)
            missing.append(p)
            continue
        # Already absolute/exists
        if os.path.exists(p):
            resolved.append(p)
            continue
        # Try roots
        hit = None
        for r in roots:
            cand = os.path.join(r, p)
            if os.path.exists(cand):
                hit = cand
                break
            cand2 = str(pathlib.Path(r) / pathlib.Path(p))
            if os.path.exists(cand2):
                hit = cand2
                break
        if hit is None:
            resolved.append(None)
            missing.append(p)
        else:
            resolved.append(hit)
    return resolved, missing


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32)


def is_npy(p: Optional[str]) -> bool:
    return (p is not None) and p.lower().endswith(".npy") and os.path.exists(p)


# -------------- Encoders --------------
@torch.no_grad()
def encode_with_imagebind(paths: List[str], device: str, batch: int) -> np.ndarray:
    assert HAS_IMAGEBIND, "ImageBind is not available"
    dev = torch.device(device)
    model = imagebind_huge(pretrained=True).to(dev).eval()
    outs = []
    for i in range(0, len(paths), batch):
        chunk = paths[i : i + batch]
        vis = load_and_transform_vision_data(chunk, dev)
        emb = model({ModalityType.VISION: vis})[ModalityType.VISION]
        outs.append(emb.cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def encode_with_clip(paths: List[str], device: str, batch: int, model_name: str = "ViT-B/32") -> np.ndarray:
    assert HAS_CLIP, "CLIP is not available"
    dev = torch.device(device)
    model, preprocess = clip.load(model_name, device=dev)
    model.eval()
    outs = []
    for i in range(0, len(paths), batch):
        chunk = paths[i : i + batch]
        imgs = [preprocess(Image.open(p).convert("RGB")).unsqueeze(0) for p in chunk]
        x = torch.cat(imgs, dim=0).to(dev)
        z = model.encode_image(x).float().cpu().numpy()
        outs.append(z)
    return np.concatenate(outs, axis=0)


def load_or_encode_items(items: List[str], prefer: Optional[str], device: str, batch: int) -> np.ndarray:
    N = len(items)
    first = items[0]
    # Case 1: a single big N x D array file
    if is_npy(first):
        try:
            arr = np.load(first)
            if arr.ndim == 2 and arr.shape[0] == N:
                print(f"[info] Detected a full-array .npy {arr.shape}; using it directly.")
                return arr.astype(np.float32)
        except Exception:
            pass
    # Case 2: all per-sample 1D .npy
    if all(is_npy(p) for p in items):
        parts = []
        for p in items:
            a = np.load(p)
            if a.ndim == 1:
                a = a[None, :]
            parts.append(a.astype(np.float32))
        return np.concatenate(parts, axis=0)
    # Case 3: encode images
    if prefer == "imagebind" and HAS_IMAGEBIND:
        print(f"[info] Encoding {N} images with ImageBind on {device} (batch={batch})")
        return encode_with_imagebind(items, device, batch).astype(np.float32)
    if prefer == "clip" and HAS_CLIP:
        print(f"[info] Encoding {N} images with CLIP ViT-B/32 on {device} (batch={batch})")
        return encode_with_clip(items, device, batch).astype(np.float32)
    # Fallback
    if HAS_IMAGEBIND:
        print(f"[info] Fallback to ImageBind on {device} (batch={batch})")
        return encode_with_imagebind(items, device, batch).astype(np.float32)
    if HAS_CLIP:
        print(f"[info] Fallback to CLIP on {device} (batch={batch})")
        return encode_with_clip(items, device, batch).astype(np.float32)
    raise RuntimeError("No image encoder available. Install 'imagebind' or 'clip'.")


def build_text_embeddings(class_texts: List[str], device: str, prefer: Optional[str], batch: int) -> np.ndarray:
    if prefer == "s2" and HAS_S2:
        print("[text] using sentence-transformers (all-mpnet-base-v2)")
        model = SentenceTransformer("all-mpnet-base-v2")
        embs = model.encode(class_texts, batch_size=batch, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    elif prefer == "clip" and HAS_CLIP:
        print("[text] using CLIP text encoder")
        dev = torch.device(device)
        model, _ = clip.load("ViT-B/32", device=dev)
        model.eval()
        all_emb = []
        for i in range(0, len(class_texts), batch):
            chunk = class_texts[i:i+batch]
            tokens = clip.tokenize(chunk).to(dev)
            with torch.no_grad():
                t = model.encode_text(tokens).float().cpu().numpy()
            all_emb.append(t)
        embs = np.concatenate(all_emb, axis=0).astype(np.float32)
    else:
        raise RuntimeError("No text encoder available. Install sentence-transformers or CLIP.")
    # L2 normalize
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return embs


def main():
    cfg = CONFIG
    csv_path = cfg["csv_path"]

    if not os.path.exists(csv_path):
        print(f"[error] CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required = ["url", "tactile", "caption", "index"]
    for c in required:
        if c not in df.columns:
            print(f"[error] CSV must contain column '{c}'. Found: {df.columns.tolist()}")
            sys.exit(1)

    N = len(df)
    print(f"[info] Rows in CSV: {N}")

    rgb_raw = df["url"].astype(str).tolist()
    tac_raw = df["tactile"].astype(str).tolist()
    captions = df["caption"].astype(str).fillna("").tolist()
    indices_raw = df["index"].tolist()

    # Cast indices to int64 if possible
    try:
        indices = np.array([int(x) for x in indices_raw], dtype=np.int64)
        print("[info] Indices cast to int64.")
    except Exception:
        indices = np.array([str(x) for x in indices_raw], dtype=object)
        print("[warn] Indices saved as strings (non-integer values detected).")

    # Resolve paths
    roots = canonical_roots(cfg)
    resolved_rgb, rgb_nf = resolve_paths(rgb_raw, roots, allow_http=cfg["allow_http_download"], cache_dir=cfg["download_cache_dir"])
    resolved_tac, tac_nf = resolve_paths(tac_raw, roots, allow_http=cfg["allow_http_download"], cache_dir=cfg["download_cache_dir"])
    if rgb_nf or tac_nf:
        print("[error] Missing files; please fix CSV or update candidate_roots. Showing up to 20 each:")
        if rgb_nf:
            print("  rgb missing:", rgb_nf[:20], "..." if len(rgb_nf) > 20 else "")
        if tac_nf:
            print("  tactile missing:", tac_nf[:20], "..." if len(tac_nf) > 20 else "")
        sys.exit(1)

    # Caption -> class id
    caption_to_cid = {}
    class_texts = []
    sample_labels = np.empty(N, dtype=np.int64)
    for i, c in enumerate(captions):
        key = c.strip()
        if key not in caption_to_cid:
            caption_to_cid[key] = len(class_texts)
            class_texts.append(key)
        sample_labels[i] = caption_to_cid[key]
    num_classes = len(class_texts)
    print(f"[info] Unique captions/classes: {num_classes}")

    # Encode rgb & tactile
    if CONFIG["prefer_image_encoder"] is None:
        print("[error] No image encoder available. Install 'imagebind' or 'clip'.")
        sys.exit(1)

    rgb_embs = load_or_encode_items(resolved_rgb, cfg["prefer_image_encoder"], cfg["device"], cfg["image_batch"]).astype(np.float32)
    tac_embs = load_or_encode_items(resolved_tac, cfg["prefer_image_encoder"], cfg["device"], cfg["image_batch"]).astype(np.float32)

    if cfg["l2_normalize"]:
        rgb_embs = l2_normalize_rows(rgb_embs)
        tac_embs = l2_normalize_rows(tac_embs)
        print("[info] L2-normalized embeddings (rgb & tactile).")

    print(f"[info] rgb_embs shape: {rgb_embs.shape}; tac_embs shape: {tac_embs.shape}")

    # --- Save outputs ---
    out_root = cfg["out_root"]
    out_rgb_dir = cfg["out_rgb_dir"]
    out_tac_dir = cfg["out_tac_dir"]
    ensure_dir(out_root); ensure_dir(out_rgb_dir); ensure_dir(out_tac_dir)

    out_rgb_emb = ensure_out(out_rgb_dir, cfg["out_all_emb"])
    out_rgb_idx = ensure_out(out_rgb_dir, cfg["out_all_idx"])
    out_tac_emb = ensure_out(out_tac_dir, cfg["out_all_emb"])
    out_tac_idx = ensure_out(out_tac_dir, cfg["out_all_idx"])

    np.save(out_rgb_emb, rgb_embs.astype(np.float32))
    np.save(out_tac_emb, tac_embs.astype(np.float32))

    if indices.dtype.kind in ("i", "u"):
        np.save(out_rgb_idx, indices.astype(np.int64))
        np.save(out_tac_idx, indices.astype(np.int64))
    else:
        np.save(out_rgb_idx, indices.astype(object))
        np.save(out_tac_idx, indices.astype(object))

    print(f"[save] rgb  embeddings -> {out_rgb_emb}")
    print(f"[save] rgb  indices    -> {out_rgb_idx}")
    print(f"[save] tac  embeddings -> {out_tac_emb}")
    print(f"[save] tac  indices    -> {out_tac_idx}")

    # Labels
    out_labels = ensure_out(out_root, cfg["out_labels"])
    np.save(out_labels, sample_labels.astype(np.int64))
    print(f"[save] per-sample labels -> {out_labels}")

    # Manifests
    out_manifest_rgb = ensure_out(out_root, cfg["out_manifest_rgb"])
    out_manifest_tac = ensure_out(out_root, cfg["out_manifest_tac"])
    with open(out_manifest_rgb, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["index", "resolved_rgb_path"])
        for idx, p in zip(indices_raw, resolved_rgb):
            w.writerow([idx, p])
    with open(out_manifest_tac, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["index", "resolved_tactile_path"])
        for idx, p in zip(indices_raw, resolved_tac):
            w.writerow([idx, p])
    print(f"[save] manifest_rgb -> {out_manifest_rgb}")
    print(f"[save] manifest_tac -> {out_manifest_tac}")

    # Optional per-sample saves
    if cfg["save_per_sample_rgb"]:
        per_dir = cfg["per_sample_rgb_dir"]; ensure_dir(per_dir)
        for i, idx in enumerate(indices):
            np.save(os.path.join(per_dir, f"{idx}_embedding.npy"), rgb_embs[i])
        print(f"[save] per-sample RGB embeddings -> {per_dir}")
    if cfg["save_per_sample_tac"]:
        per_dir = cfg["per_sample_tac_dir"]; ensure_dir(per_dir)
        for i, idx in enumerate(indices):
            np.save(os.path.join(per_dir, f"{idx}_embedding.npy"), tac_embs[i])
        print(f"[save] per-sample tactile embeddings -> {per_dir}")

    # Text embeddings for unique captions
    try:
        text_embs = build_text_embeddings(class_texts, cfg["device"], cfg["prefer_text_encoder"], batch=cfg["text_batch"])
        out_text = ensure_out(out_root, cfg["out_text_embs"])
        np.save(out_text, text_embs.astype(np.float32))
        print(f"[save] text embeddings -> {out_text} (shape {text_embs.shape})")
    except Exception as e:
        print("[warn] failed to build text embeddings:", e)
        print("       You can later build text embeddings into:", ensure_out(out_root, cfg["out_text_embs"]))

    print("\n[done] Files generated:")
    print(" ", out_rgb_emb)
    print(" ", out_rgb_idx)
    print(" ", out_tac_emb)
    print(" ", out_tac_idx)
    print(" ", out_labels)
    print(" ", out_manifest_rgb)
    print(" ", out_manifest_tac)
    text_p = ensure_out(out_root, cfg["out_text_embs"])  # path only for display
    if os.path.exists(text_p):
        print(" ", text_p)


if __name__ == "__main__":
    main()
