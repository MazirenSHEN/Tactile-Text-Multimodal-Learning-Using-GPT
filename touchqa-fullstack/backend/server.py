# -*- coding: utf-8 -*-
# backend/server.py
# Avoid OpenMP conflicts during development (faiss/mkl/torch)
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import tempfile
from typing import Optional, Tuple, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Optional: load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import pandas as pd

# --- Import your integrated model ---
try:
    from TactileQASystem_integrated import TouchQAModel  # directly use your class
except Exception as e:
    raise RuntimeError(
        "Failed to import TouchQAModel. Please ensure TactileQASystem_integrated.py is in backend/ and the class name is correct."
    ) from e


def getenv(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    return "" if v is None else v


# --- Paths/params (overridable via .env) ---
TACTILE_EMB_DIR  = getenv("TACTILE_EMB_DIR", "embeddings/embeddings_tac")
RGB_EMB_DIR      = getenv("RGB_EMB_DIR", "embeddings/embeddings_rgb")
PROJECTOR_PATH   = getenv("PROJECTOR_PATH", "tac_projector_vit5p_best.pt")  # leave empty to disable
CAPTION_CSV      = getenv("CAPTION_CSV", "data/ssvtp/new_train.csv")
TAC_IMG_DIR      = getenv("TAC_IMG_DIR", "data/ssvtp/images_tac")
RGB_IMG_DIR      = getenv("RGB_IMG_DIR", "data/ssvtp/images_rgb")
MIN_COS          = float(getenv("MIN_COS", "0.0"))
OVERFETCH_MUL    = int(getenv("OVERFETCH_MUL", "2"))
OVERFETCH_BIAS   = int(getenv("OVERFETCH_BIAS", "5"))
NORMALIZE_BEFORE = getenv("NORMALIZE_BEFORE_PROJECTOR", "false").lower() in ("1","true","yes")


# --- CSV self-healing: ensure columns 'index' and 'caption' exist ---
def normalize_caption_csv(path: str) -> str:
    try:
        df = pd.read_csv(path)
    except Exception:
        return path
    df.columns = [c.lower() for c in df.columns]
    if "index" not in df.columns:
        for cand in ["id", "image_id", "img_id", "tac_id", "rgb_id"]:
            if cand in df.columns:
                df["index"] = df[cand]
                break
        else:
            for cand in ["path", "tac_path", "rgb_path", "filename", "file"]:
                if cand in df.columns:
                    df["index"] = df[cand].astype(str).str.extract(r"(\d+)").iloc[:, 0]
                    break
    if "index" not in df.columns:
        df["index"] = ""
    if "caption" not in df.columns:
        for c in ["text", "desc", "description", "label"]:
            if c in df.columns:
                df = df.rename(columns={c: "caption"})
                break
        else:
            df["caption"] = ""
    df["index"] = df["index"].astype(str).fillna("")
    df["caption"] = df["caption"].astype(str).fillna("")
    out_path = os.path.join(os.path.dirname(path), "_normalized_train.csv")
    df[["index", "caption"]].to_csv(out_path, index=False)
    return out_path


# --- Mode normalization (only for input-shape detection; unrelated to intent) ---
def normalize_mode(s: Optional[str]) -> str:
    if not s:
        return "auto"
    t = s.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "auto": "auto", "default": "auto",
        "image_and_text": "image_and_text", "image+text": "image_and_text", "img_text": "image_and_text",
        "image_only": "image_only", "imageonly": "image_only", "img": "image_only",
        "text_only": "text_only", "textonly": "text_only", "text": "text_only",
    }
    return mapping.get(t, "auto")


# --- FastAPI ---
app = FastAPI(title="TouchQA Backend", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for RGB reference images (if it exists)
if os.path.isdir(RGB_IMG_DIR):
    app.mount("/rgb", StaticFiles(directory=RGB_IMG_DIR), name="rgb")


MODEL: Optional[TouchQAModel] = None


def load_model() -> TouchQAModel:
    global MODEL
    if MODEL is None:
        csv_path = normalize_caption_csv(CAPTION_CSV)
        MODEL = TouchQAModel(
            tactile_emb_dir=TACTILE_EMB_DIR,
            rgb_emb_dir=RGB_EMB_DIR,
            projector_path=PROJECTOR_PATH,
            caption_csv=csv_path,
            tactile_img_dir=TAC_IMG_DIR,
            rgb_img_dir=RGB_IMG_DIR,
            projector_sensor_id=None,
            min_cos=MIN_COS,
            overfetch_mul=OVERFETCH_MUL,
            overfetch_bias=OVERFETCH_BIAS,
            normalize_before_projector=NORMALIZE_BEFORE,
        )
    return MODEL


@app.get("/api/health")
def health():
    m = load_model()
    proj = bool(getattr(m, "projector", None))
    return {"ok": True, "projector_loaded": proj}


@app.get("/api/config")
def config():
    return {
        "TACTILE_EMB_DIR": TACTILE_EMB_DIR,
        "RGB_EMB_DIR": RGB_EMB_DIR,
        "PROJECTOR_PATH": PROJECTOR_PATH,
        "CAPTION_CSV": CAPTION_CSV,
        "TAC_IMG_DIR": TAC_IMG_DIR,
        "RGB_IMG_DIR": RGB_IMG_DIR,
    }


# --- Auto-detect RGB/Tac and, for Tac, retrieve nearest RGB neighbors ---
def detect_modality_and_neighbors(model: TouchQAModel, image_path: str, intent: str) -> tuple[str, list]:
    """
    Returns: (modality, neighbors)
    modality ∈ {'rgb', 'tac'}
    neighbors is non-empty only for 'tac' (list of dicts: {id, score, caption, rgb_url})
    """
    neighbors: List[dict] = []
    modality = "tac"  # default conservatively treat as tac

    try:
        emb_raw = model.extract_raw_embedding(image_path)  # your method
        if model.is_rgb(emb_raw):
            modality = "rgb"
        else:
            modality = "tac"
    except Exception:
        # If feature extraction or classifier fails, fall back to tac (conservative)
        modality = "tac"

    if modality == "tac":
        # Tac branch only: perform Tac→RGB retrieval (your logic: map to RGB space if projector exists)
        try:
            topk = getattr(model, "intent_topk", {}).get(intent, 2)
            if getattr(model, "projector", None) is not None:
                q_vec = model.apply_projector_to_vector(emb_raw)  # type: ignore
            else:
                q_vec = emb_raw
            rgb_index = getattr(model, "rgb_index", None)
            if rgb_index is not None:
                pairs = model._search_ids(rgb_index, q_vec, topk)  # type: ignore
                idx2cap = getattr(model, "idx2caption", {}) or {}
                for rid, score in pairs:
                    cap = idx2cap.get(str(rid), "<unknown>")
                    img_path = os.path.join(RGB_IMG_DIR, f"image_{rid}_rgb.jpg")
                    neighbors.append({
                        "id": int(rid),
                        "score": float(score),
                        "caption": cap,
                        "rgb_url": f"/rgb/image_{rid}_rgb.jpg" if os.path.exists(img_path) else None,
                    })
        except Exception:
            # If retrieval fails, return empty neighbors; do not block answering
            neighbors = []

    return modality, neighbors


@app.post("/api/answer")
async def api_answer(
    image: Optional[UploadFile] = File(None),
    question: Optional[str] = Form(None),
    mode: str = Form("auto"),
):
    """
    mode describes the input shape only:
      - auto: infer from submitted fields
      - image_only: image only
      - text_only: question only
      - image_and_text: image + question

    Intent classification is independent of mode, via your classify_intent:
      -> property / comparison / judgement / other
    """
    model = load_model()
    mode = normalize_mode(mode)

    # Infer input shape from submitted fields if 'auto'
    if mode == "auto":
        if image and question:
            mode = "image_and_text"
        elif image and not question:
            mode = "image_only"
        elif question and not image:
            mode = "text_only"
        else:
            return {"answer": "[Error] No input provided.", "mode": "none", "intent": "other", "modality": "unknown", "neighbors": []}

    neighbors: List[dict] = []
    modality = "unknown"
    intent = "other"

    # First do intent classification (if no question, default to 'other')
    try:
        if question:
            intent = model.classify_intent(question)  # property | comparison | judgement | other
    except Exception:
        intent = "other"

    tmp_path = None

    try:
        # For modes that need an image: save a temp file and auto-detect RGB/Tac; for Tac, retrieve neighbors
        if mode in ("image_and_text", "image_only"):
            suffix = os.path.splitext(getattr(image, "filename", "") or "")[-1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                content = await image.read()  # type: ignore[attr-defined]
                tf.write(content)
                tmp_path = tf.name

            modality, neighbors = detect_modality_and_neighbors(model, tmp_path, intent)

        # —— Delegate actual answering to your model —— #
        def answer_minimal(image_path, q_text: str) -> str:
            # Prefer the minimal signature; if different, fall back to the version with history=None
            try:
                return model.answer(image_path=image_path, user_query=q_text)  # type: ignore[call-arg]
            except TypeError:
                pass
            try:
                return model.answer(image_path=image_path, user_query=q_text, history=None)  # type: ignore[call-arg]
            except Exception as e:
                return f"[Answer Fallback] {e}"

        if mode == "image_and_text":
            answer = answer_minimal(tmp_path, question or "")
        elif mode == "image_only":
            if hasattr(model, "answer_image_only"):
                answer = model.answer_image_only(tmp_path)  # type: ignore[attr-defined]
            else:
                # Let your answer route internally (RGB: no retrieval; Tac: perform Tac→RGB and build message with references)
                answer = answer_minimal(tmp_path, "")
        elif mode == "text_only":
            if hasattr(model, "answer_text_only"):
                answer = model.answer_text_only(question or "")  # type: ignore[attr-defined]
            else:
                try:
                    answer = model.answer(image_path=None, user_query=question or "", history=None)  # type: ignore[arg-type]
                except Exception as e:
                    answer = f"[Text-only Fallback] {e}"
        else:
            return {"answer": "[Error] Unsupported mode.", "mode": mode, "intent": intent, "modality": modality, "neighbors": []}

        return {
            "answer": answer,
            "mode": mode,
            "intent": intent,        # property / comparison / judgement / other
            "modality": modality,    # rgb / tac / unknown
            "neighbors": neighbors,  # non-empty only for tac (determined by backend retrieval); always empty for rgb
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
