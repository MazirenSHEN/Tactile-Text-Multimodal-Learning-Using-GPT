# build_test_embeddings.py
"""
Convert test.csv -> numpy embeddings for retrieval & zero-shot testing.

Input CSV must contain columns: at least ['tactile', 'rgb', 'caption', 'index']
Optional column: 'sensor_id' (will be saved as test_sensor_ids.npy)

Outputs (saved under embeddings/):
 - embeddings/test_tac.npy, embeddings/test_rgb.npy
 - embeddings/test_tac_idx.npy, embeddings/test_rgb_idx.npy
 - embeddings/test_labels.npy  (per-sample class id)
 - embeddings/text_embs.npy    (class-level text embeddings, normalized)
 - optionally embeddings/test_sensor_ids.npy
"""
import os
import sys
import json
import pathlib
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from PIL import Image

# ---------------- Backends availability ----------------
HAS_IMAGEBIND = False
try:
    from imagebind.models.imagebind_model import imagebind_huge, ModalityType
    from imagebind.data import load_and_transform_vision_data
    HAS_IMAGEBIND = True
except Exception:
    HAS_IMAGEBIND = False

HAS_CLIP = False
try:
    import clip
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
    "csv_path": "../data/test/updated_test.csv",

    "tactile_col_candidates": ["tactile", "tac_path", "tactile_path"],
    "rgb_col_candidates":     ["rgb", "rgb_path", "image_path"],
    "caption_col_candidates": ["caption", "text"],
    "index_col_candidates":   ["index", "id", "image_id"],
    "sensor_col_candidates":  ["sensor_id", "sensor"],

    # Output directory + file names (will be joined as out_dir/filename if not absolute)
    "out_dir":        "embeddings/test2",
    "out_tac":        "test_tac.npy",
    "out_rgb":        "test_rgb.npy",
    "out_tac_idx":    "test_tac_idx.npy",
    "out_rgb_idx":    "test_rgb_idx.npy",
    "out_labels":     "test_labels.npy",
    "out_text_embs":  "text_embs.npy",
    "out_sensor_ids": "test_sensor_ids.npy",

    # Device & batch
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_batch": 128,
    "text_batch":  64,

    # Preferred encoders (auto if available)
    "prefer_image_encoder": "imagebind" if HAS_IMAGEBIND else ("clip" if HAS_CLIP else None),
    "prefer_text_encoder":  "s2" if HAS_S2 else ("clip" if HAS_CLIP else None),
}
# ------------------------------------------------------


def find_column(df, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def out_path(root: str, name: str) -> str:
    """Join out_dir and filename if 'name' is not absolute and has no parent dir."""
    if os.path.isabs(name):
        return name
    if os.path.dirname(name):
        # name already has a subdir like "embeddings/foo.npy"
        os.makedirs(os.path.dirname(name), exist_ok=True)
        return name
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, name)


# --------- Path resolution: map relative CSV paths to actual existing files ----------
def resolve_paths(items: List[str], roots: List[str]):
    """
    For each path in the CSV (which may be relative), try joining with several candidate root directories
    and find an existing file. Returns: resolved (list[str or None]), not_found (list[str])
    """
    resolved = []
    not_found = []
    for p in items:
        p = ("" if p is None else str(p).strip())
        if p == "":
            resolved.append(None)
            not_found.append(p)
            continue
        # direct path exists
        if os.path.exists(p):
            resolved.append(p)
            continue
        # try candidate roots
        hit = None
        for r in roots:
            cand = os.path.join(r, p)
            if os.path.exists(cand):
                hit = cand
                break
            cand2 = str(pathlib.Path(r) / pathlib.Path(p))  # handle different path separators
            if os.path.exists(cand2):
                hit = cand2
                break
        if hit is None:
            resolved.append(None)
            not_found.append(p)
        else:
            resolved.append(hit)
    return resolved, not_found


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ---------- Image encoders ----------
@torch.no_grad()
def encode_images_imagebind(paths: List[str], device: str, batch: int = 64) -> np.ndarray:
    assert HAS_IMAGEBIND, "imagebind not available"
    dev = torch.device(device)
    model = imagebind_huge(pretrained=True).to(dev).eval()
    outs = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i+batch]
        vis = load_and_transform_vision_data(chunk, dev)
        out = model({ModalityType.VISION: vis})[ModalityType.VISION].cpu().numpy()
        outs.append(out)
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def encode_images_clip(paths: List[str], device: str, batch: int = 64, model_name: str = "ViT-B/32") -> np.ndarray:
    assert HAS_CLIP, "clip not available"
    dev = torch.device(device)
    model, preprocess = clip.load(model_name, device=dev)
    model.eval()
    outs = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i+batch]
        imgs = [preprocess(load_image(p)).unsqueeze(0) for p in chunk]
        x = torch.cat(imgs, dim=0).to(dev)
        z = model.encode_image(x).float().cpu().numpy()
        outs.append(z)
    return np.concatenate(outs, axis=0)


# ---------- Text encoder ----------
def build_text_embeddings(class_texts: List[str], device: str, prefer="s2", batch:int=64) -> np.ndarray:
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

    # Candidate root directories (search priority)
    csv_dir = os.path.dirname(csv_path) or "."
    IMAGE_ROOTS = [
        ".",                 # current working dir
        csv_dir,             # CSV directory
        "data/ssvtp",        # common dataset root
    ]
    # dedupe & normalize
    IMAGE_ROOTS = list(dict.fromkeys([str(pathlib.Path(r)) for r in IMAGE_ROOTS]))

    # Read CSV and identify columns
    df = pd.read_csv(csv_path)
    tac_col = find_column(df, cfg["tactile_col_candidates"])
    rgb_col = find_column(df, cfg["rgb_col_candidates"])
    cap_col = find_column(df, cfg["caption_col_candidates"])
    idx_col = find_column(df, cfg["index_col_candidates"])
    sensor_col = find_column(df, cfg["sensor_col_candidates"])

    if None in (tac_col, rgb_col, cap_col, idx_col):
        print("[error] required columns not detected. CSV columns:", df.columns.tolist())
        sys.exit(1)

    N = len(df)
    print(f"[info] rows in CSV: {N}")

    tac_raw = df[tac_col].astype(str).tolist()
    rgb_raw = df[rgb_col].astype(str).tolist()
    captions = df[cap_col].astype(str).fillna("").tolist()
    indices  = df[idx_col].tolist()

    sensor_ids = None
    if sensor_col:
        try:
            sensor_ids = df[sensor_col].fillna(-1).astype(int).to_numpy()
            print(f"[info] sensor_id column found, unique sensors: {np.unique(sensor_ids)}")
        except Exception:
            sensor_ids = None

    # Resolve paths (map CSV relative paths to actual files)
    tac_items, tac_nf = resolve_paths(tac_raw, IMAGE_ROOTS)
    rgb_items, rgb_nf = resolve_paths(rgb_raw, IMAGE_ROOTS)
    if tac_nf or rgb_nf:
        print("[error] The following paths could not be found (check CSV or add the correct root to IMAGE_ROOTS):")
        if tac_nf:
            print("  tactile missing (showing up to 20):", tac_nf[:20], "..." if len(tac_nf) > 20 else "")
        if rgb_nf:
            print("  rgb     missing (showing up to 20):", rgb_nf[:20], "..." if len(rgb_nf) > 20 else "")
        sys.exit(1)

    # Build caption -> class id mapping
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
    print(f"[info] found {num_classes} unique captions/classes.")

    # Save indices (used for retrieval/eval pairing)
    out_dir = cfg["out_dir"]
    tac_idx_path = out_path(out_dir, cfg["out_tac_idx"])
    rgb_idx_path = out_path(out_dir, cfg["out_rgb_idx"])
    np.save(tac_idx_path, np.array(indices))
    np.save(rgb_idx_path, np.array(indices))
    print(f"[save] tac_idx -> {tac_idx_path}")
    print(f"[save] rgb_idx -> {rgb_idx_path}")

    # Decide whether inputs are .npy bundles or image paths; if .npy load directly, else run image encoder
    def is_npy(p: str) -> bool:
        return (p is not None) and p.lower().endswith(".npy") and os.path.exists(p)

    def load_or_encode_items(items: List[str], kind: str) -> np.ndarray:
        # If the first item is a full N x D .npy array, use it directly
        first = items[0]
        if is_npy(first):
            try:
                arr = np.load(first)
                if arr.ndim == 2 and arr.shape[0] == N:
                    print(f"[info] {kind}: detected a full-array .npy {arr.shape}, use it directly.")
                    return arr.astype(np.float32)
            except Exception:
                pass

        # If all are per-sample .npy files, stack them
        if all(is_npy(x) for x in items):
            parts = []
            for p in items:
                a = np.load(p)
                if a.ndim == 1:
                    a = a[None, :]
                parts.append(a.astype(np.float32))
            stacked = np.concatenate(parts, axis=0)
            if stacked.shape[0] != N:
                print(f"[warn] stacked {kind} rows {stacked.shape[0]} != CSV N {N}")
            return stacked

        # Otherwise run an image encoder
        print(f"[info] encoding {len(items)} {kind} images with '{cfg['prefer_image_encoder']}' on {cfg['device']}")
        if cfg["prefer_image_encoder"] == "imagebind" and HAS_IMAGEBIND:
            return encode_images_imagebind(items, cfg["device"], batch=cfg["image_batch"]).astype(np.float32)
        elif cfg["prefer_image_encoder"] == "clip" and HAS_CLIP:
            return encode_images_clip(items, cfg["device"], batch=cfg["image_batch"]).astype(np.float32)
        else:
            # fallback order
            if HAS_IMAGEBIND:
                return encode_images_imagebind(items, cfg["device"], batch=cfg["image_batch"]).astype(np.float32)
            if HAS_CLIP:
                return encode_images_clip(items, cfg["device"], batch=cfg["image_batch"]).astype(np.float32)
            raise RuntimeError("No image encoder available. Install imagebind or CLIP.")

    tac_embs = load_or_encode_items(tac_items, "tactile")
    rgb_embs = load_or_encode_items(rgb_items, "rgb")
    print(f"[info] tac_embs shape: {tac_embs.shape}, rgb_embs shape: {rgb_embs.shape}")

    # Save embeddings
    out_tac = out_path(out_dir, cfg["out_tac"])
    out_rgb = out_path(out_dir, cfg["out_rgb"])
    np.save(out_tac, tac_embs.astype(np.float32))
    np.save(out_rgb, rgb_embs.astype(np.float32))
    print(f"[save] tactile embeddings -> {out_tac}")
    print(f"[save] rgb     embeddings -> {out_rgb}")

    # Save labels / sensor_ids
    out_labels = out_path(out_dir, cfg["out_labels"])
    np.save(out_labels, sample_labels)
    print(f"[save] per-sample labels -> {out_labels}")

    if sensor_ids is not None:
        out_sensor = out_path(out_dir, cfg["out_sensor_ids"])
        np.save(out_sensor, sensor_ids.astype(np.int64))
        print(f"[save] sensor ids -> {out_sensor}")

    # Build and save text class embeddings
    try:
        text_embs = build_text_embeddings(class_texts, cfg["device"], prefer=cfg["prefer_text_encoder"], batch=cfg["text_batch"])
        out_text = out_path(out_dir, cfg["out_text_embs"])
        np.save(out_text, text_embs.astype(np.float32))
        print(f"[save] text embeddings -> {out_text} (shape {text_embs.shape})")
    except Exception as e:
        print("[warn] failed to build text embeddings:", e)
        print("You can later build text embeddings into:", out_path(out_dir, cfg['out_text_embs']))

    print("\n[done] Files generated:")
    print(" ", out_tac)
    print(" ", out_rgb)
    print(" ", tac_idx_path)
    print(" ", rgb_idx_path)
    print(" ", out_labels)
    text_p = out_path(out_dir, cfg["out_text_embs"])
    if os.path.exists(text_p):
        print(" ", text_p)
    if sensor_ids is not None:
        print(" ", out_path(out_dir, cfg["out_sensor_ids"]))


if __name__ == "__main__":
    main()
