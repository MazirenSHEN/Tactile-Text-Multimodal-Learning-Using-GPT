# TactileQASystem_integrated.py  (RESTORED for ablations)
"""
This version restores the functions/attributes the ablation scripts expect:
- Loads BOTH tactile and RGB embeddings + builds FAISS IDMap indices.
- Provides: tac_raw, tac_emb, tac_ids, tac_index
- Provides (when projector exists): tac_proj (projected tac vectors), tac_proj_index
- Provides: rgb_emb (L2), rgb_ids, rgb_index
- Methods:
    * extract_raw_embedding(image_path)  -> raw ImageBind vector (float32)
    * apply_projector_to_vector(vec)     -> projected vec (float32, L2)
    * project_all_tactile(batch=256)     -> NxD tactile-projected (float32, L2) + caches tac_proj/_index
    * get_arrays_for_eval()              -> (tac_raw, tac_ids, rgb_emb, rgb_ids)

Also keeps:
- _search_ids(index, q_vec, topk)
- answer() for QA (tac→rgb-neighbors label prompting)

Safe fallbacks:
- If OpenAI / ImageBind are not installed, functions that need them will raise clear errors
  or return conservative defaults (e.g., classify_intent -> "other").
"""

from __future__ import annotations

import os
import base64
from functools import lru_cache
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import faiss

# -------- Optional deps (graceful fallback) --------
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore

try:
    from imagebind.models.imagebind_model import imagebind_huge, ModalityType
    from imagebind.data import load_and_transform_vision_data
    _HAS_IMAGEBIND = True
except Exception:
    _HAS_IMAGEBIND = False
    imagebind_huge = None  # type: ignore
    ModalityType = None    # type: ignore
    load_and_transform_vision_data = None  # type: ignore

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# =============================
# ViTProjector (same structure as training)
# =============================
class ViTProjector(nn.Module):
    def __init__(
        self,
        emb_dim: int = 1024,
        num_tokens: int = 16,
        dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_sensors: int = 0,
        sensor_token_len: int = 5,
        learnable_tau: bool = True,
        init_tau: float = 0.07,
        residual: bool = True,
        layerscale_init: float = 1e-3,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.emb_dim = emb_dim
        self.sensor_token_len = sensor_token_len if num_sensors > 0 else 0
        self.num_sensors = num_sensors
        self.residual = residual

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

        # gamma vector (len = emb_dim) to be robust across ckpt diffs
        self.gamma = nn.Parameter(torch.ones(emb_dim) * layerscale_init) if residual else None

        if learnable_tau:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau)))
            self.tau = None
        else:
            self.register_buffer("tau", torch.tensor(init_tau))
            self.log_tau = None

        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def get_tau(self):
        if self.log_tau is not None:
            return torch.exp(self.log_tau)
        return self.tau

    def forward(self, x: torch.Tensor, sensor_ids: torch.Tensor | None = None) -> torch.Tensor:
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
        delta = self.head(cls_out)
        if self.residual:
            y = x + (delta * self.gamma if self.gamma is not None else delta)
        else:
            y = delta
        return y


def load_vit_projector(projector_path: str, device: str = "cpu", default_emb_dim: int = 1024) -> nn.Module:
    """Robust loader for trained ViT projector checkpoints.
    Supports {'state_dict','config'} dicts or plain state_dict files.
    AUTO-ALIGNS residual flag with the checkpoint to avoid 'gamma' mismatch.
    """
    ckpt = torch.load(projector_path, map_location=device)

    def _build(cfg, residual_flag):
        return ViTProjector(
            emb_dim=cfg.get('emb_dim', default_emb_dim),
            num_tokens=cfg.get('num_tokens', 16),
            dim=cfg.get('dim', 256),
            depth=cfg.get('depth', 4),
            heads=cfg.get('heads', 8),
            num_sensors=cfg.get('num_sensors', 0),
            sensor_token_len=cfg.get('sensor_token_len', 5),
            learnable_tau=cfg.get('learnable_tau', True),
            init_tau=cfg.get('init_tau', 0.07),
            residual=residual_flag,
            layerscale_init=cfg.get('layerscale_init', 1e-3),
        )

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        cfg = ckpt.get('config', {})
        residual_cfg = bool(cfg.get('residual', True))
        model = _build(cfg, residual_cfg)
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)

        # If residual=True but ckpt lacks gamma (or dims mismatch), retry with residual=False
        need_toggle = ('gamma' in missing) or any('gamma' in u for u in unexpected)
        if need_toggle and residual_cfg:
            model = _build(cfg, residual_flag=False)
            missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        if missing or unexpected:
            print(f"[load_vit_projector] missing={missing}, unexpected={unexpected}")
        return model.to(device).eval()

    # fallback: assume plain state_dict (conservative residual=False)
    model = ViTProjector(emb_dim=default_emb_dim, residual=False)
    model.load_state_dict(ckpt, strict=False)
    return model.to(device).eval()


# =============================
# TouchQA model (now with full arrays/indices for ablations)
# =============================
class TouchQAModel:
    def __init__(
        self,
        tactile_emb_dir,   # expects all_embeddings.npy + all_indices.npy
        rgb_emb_dir,       # expects all_embeddings.npy + all_indices.npy
        projector_path,
        caption_csv,
        emb_dim: int = 1024,
        tactile_img_dir: str = "data/ssvtp/images_tac",
        rgb_img_dir: str = "data/ssvtp/images_rgb",
        projector_sensor_id: int | None = None,
        # retrieval config
        min_cos: float = 0.0,
        overfetch_mul: int = 2,
        overfetch_bias: int = 5,
        # projector input pre-norm (testvit-style = False)
        normalize_before_projector: bool = False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.emb_dim = int(emb_dim)
        self.min_cos = float(min_cos)
        self.overfetch_mul = int(overfetch_mul)
        self.overfetch_bias = int(overfetch_bias)
        self.normalize_before_projector = bool(normalize_before_projector)
        self.projector_sensor_id = projector_sensor_id

        # Optional vision encoder (only needed for query_from='image')
        self.model = None
        if _HAS_IMAGEBIND:
            try:
                self.model = imagebind_huge(pretrained=True).to(self.device).eval()
            except Exception as e:
                print(f"[TouchQAModel] ImageBind init failed: {e}")

        # Optional OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if (_HAS_OPENAI and os.getenv("OPENAI_API_KEY")) else None

        # Optional RGB/Tactile router（不存在则默认走 tactile 分支）
        self.classifier = None
        clf_path = "tactile_rgb_classifier.pkl"
        if os.path.exists(clf_path):
            try:
                self.classifier = joblib.load(clf_path)
            except Exception as e:
                print(f"[TouchQAModel] Warning: failed to load classifier ({e}); routing will default to tactile branch).")

        # ----- Load RGB embeddings/index -----
        def _f32c(x: np.ndarray, ensure_writable: bool = True) -> np.ndarray:
            y = np.asarray(x, dtype=np.float32)
            if ensure_writable and (not y.flags['WRITEABLE'] or not y.flags['C_CONTIGUOUS']):
                y = np.array(y, dtype=np.float32, order='C', copy=True)
            elif not y.flags['C_CONTIGUOUS']:
                y = np.ascontiguousarray(y)
            return y

        # RGB
        rgb_raw = np.load(os.path.join(rgb_emb_dir, "all_embeddings.npy"), mmap_mode='r')
        self.rgb_emb = _f32c(rgb_raw, ensure_writable=True)
        self.rgb_ids = np.load(os.path.join(rgb_emb_dir, "all_indices.npy")).astype("int64")
        assert self.rgb_emb.shape[1] == self.emb_dim, f"rgb emb dim {self.rgb_emb.shape[1]} != {self.emb_dim}"
        faiss.normalize_L2(self.rgb_emb)
        self.rgb_index = self._build_idmap_index(self.rgb_emb, self.rgb_ids)

        # Tactile
        tac_raw = np.load(os.path.join(tactile_emb_dir, "all_embeddings.npy"), mmap_mode='r')
        self.tac_raw = _f32c(tac_raw, ensure_writable=True)                  # NOT L2
        self.tac_ids = np.load(os.path.join(tactile_emb_dir, "all_indices.npy")).astype("int64")
        assert self.tac_raw.shape[1] == self.emb_dim, f"tac emb dim {self.tac_raw.shape[1]} != {self.emb_dim}"
        self.tac_emb = self.tac_raw.astype("float32").copy(order="C")
        faiss.normalize_L2(self.tac_emb)
        self.tac_index = self._build_idmap_index(self.tac_emb, self.tac_ids)

        # ----- Projector (optional, used to map tactile->rgb space) -----
        self.projector: Optional[nn.Module] = None
        if projector_path:
            try:
                self.projector = load_vit_projector(projector_path, device=self.device, default_emb_dim=self.emb_dim).eval()
            except Exception as e:
                print(f"[TouchQAModel] projector load failed: {e}")
                self.projector = None

        # Precompute a tac-projected index if projector exists (useful for FAISS eval path)
        self.tac_proj = None
        self.tac_proj_index = None
        if self.projector is not None:
            try:
                self.tac_proj = self.project_all_tactile(batch=256)  # L2
                self.tac_proj_index = self._build_idmap_index(self.tac_proj, self.tac_ids)
            except Exception as e:
                print(f"[TouchQAModel] tac_proj precompute failed: {e}")
                self.tac_proj = None
                self.tac_proj_index = None

        # Captions table: index -> caption (string or numeric id key)
        self.idx2caption = {}
        self.df = None
        if caption_csv and os.path.exists(caption_csv):
            try:
                df = pd.read_csv(caption_csv, dtype={"index": str})
            except Exception:
                df = pd.read_csv(caption_csv)
                if "index" not in df.columns:
                    # Heuristic: try to create an 'index' column if missing
                    for cand in ("id", "img_id", "image_id"):
                        if cand in df.columns:
                            df["index"] = df[cand].astype(str); break
                df["index"] = df["index"].astype(str) if "index" in df.columns else df.iloc[:,0].astype(str)
            self.df = df
            key_str = df["index"].astype(str)
            self.idx2caption.update(dict(zip(key_str, df.get("caption", pd.Series([""]*len(df))).astype(str))))

            # Also index by numeric id if present inside the string
            key_num = key_str.str.extract(r"(\d+)", expand=False)
            for k, cap in zip(key_num, df.get("caption", pd.Series([""]*len(df))).astype(str)):
                if pd.notna(k) and k not in self.idx2caption:
                    self.idx2caption[k] = cap

        # roots
        self.tactile_img_dir = tactile_img_dir
        self.rgb_img_dir = rgb_img_dir

        # Intent templates and dynamic top-k (labels from neighbors)
        self.intent_prompt_templates = {
            "property": "Please describe the tactile perception of the object based on the following reference sample labels: {captions}.",
            "comparison": "Compare the tactile features of the object with these reference sample labels: {captions}.",
            "judgement": "Given these tactile cues: {captions}, infer the likely material or physical state of the object.",
            "other": "Based on the tactile reference labels: {captions}, answer the following user question.",
        }
        self.intent_topk = {"property": 1, "comparison": 2, "judgement": 2, "other": 2}

    # ---------- Index utils ----------
    def _build_idmap_index(self, xb: np.ndarray, ids: np.ndarray) -> faiss.Index:
        """Build IndexFlatIP + IDMap2, add_with_ids using real IDs."""
        base = faiss.IndexFlatIP(self.emb_dim)
        index = faiss.IndexIDMap2(base)
        index.add_with_ids(np.ascontiguousarray(xb.astype("float32")), ids.astype('int64'))
        return index

    # ---------- Embedding utils ----------
    @torch.no_grad()
    def extract_raw_embedding(self, image_path: str) -> np.ndarray:
        """One ImageBind forward to get a vector. Controls L2-before-projector via flag."""
        if not _HAS_IMAGEBIND or self.model is None:
            raise RuntimeError("ImageBind not available. Install imagebind or use query_from='tacdb'.")
        vis = load_and_transform_vision_data([image_path], self.device)
        emb = self.model({ModalityType.VISION: vis})[ModalityType.VISION]
        emb = emb.detach().to("cpu").numpy()[0].astype("float32")
        emb = np.ascontiguousarray(emb)
        if self.normalize_before_projector:
            faiss.normalize_L2(emb.reshape(1, -1))
        return emb

    @torch.no_grad()
    def apply_projector_to_vector(self, emb_raw: np.ndarray) -> np.ndarray:
        """Single-vector projector forward; output will be L2-normalized."""
        assert self.projector is not None, "projector is not loaded"
        x = torch.from_numpy(np.ascontiguousarray(emb_raw.astype("float32"))).unsqueeze(0).to(self.device)
        if self.projector_sensor_id is not None:
            sid = torch.tensor([self.projector_sensor_id], device=self.device, dtype=torch.long)
            y = self.projector(x, sensor_ids=sid)
        else:
            y = self.projector(x)
        y = y.detach().to("cpu").numpy()[0].astype("float32")
        y = np.ascontiguousarray(y)
        faiss.normalize_L2(y.reshape(1, -1))
        return y

    @torch.no_grad()
    def project_all_tactile(self, batch: int = 256) -> np.ndarray:
        """Project ALL tactile raw embeddings via projector. Returns L2-normalized matrix [N,d]."""
        assert self.projector is not None, "projector is not loaded"
        X = self.tac_raw
        N = X.shape[0]
        out = np.empty_like(X, dtype="float32")
        b = int(max(1, batch))
        for i in range(0, N, b):
            x = torch.from_numpy(np.ascontiguousarray(X[i:i+b])).float().to(self.device)
            if self.projector_sensor_id is not None:
                sid = torch.full((x.size(0),), int(self.projector_sensor_id), device=self.device, dtype=torch.long)
                y = self.projector(x, sensor_ids=sid)
            else:
                y = self.projector(x)
            y = y.detach().to("cpu").numpy().astype("float32")
            out[i:i+b] = y
        faiss.normalize_L2(out)
        return out

    # ---------- Retrieval helpers ----------
    def _search_ids(self, index: faiss.Index, q: np.ndarray, topk: int) -> List[Tuple[int, float]]:
        """Return list of (real_id, score). q must be 1D float32. Overfetch + min_cos filtering."""
        q = np.ascontiguousarray(q.reshape(1, -1).astype("float32", copy=False))
        faiss.normalize_L2(q)
        k = max(int(topk) * self.overfetch_mul, int(topk) + self.overfetch_bias)
        D, I = index.search(q, k)
        res: List[Tuple[int, float]] = []
        used = set()
        for id_, s in zip(I[0].tolist(), D[0].tolist()):
            if id_ == -1 or s < self.min_cos or id_ in used:
                continue
            used.add(id_)
            res.append((int(id_), float(s)))
            if len(res) >= topk:
                break
        return res

    # ---------- Eval-friendly arrays ----------
    def get_arrays_for_eval(self):
        """Returns (tac_raw, tac_ids, rgb_emb, rgb_ids). tac_raw is NOT L2; rgb_emb IS L2."""
        return (self.tac_raw, self.tac_ids, self.rgb_emb, self.rgb_ids)

    # ---------- Routing / Messaging / QA ----------
    def is_rgb(self, emb_normed: np.ndarray) -> bool:
        # If classifier is unavailable, default to tactile branch
        if self.classifier is None:
            return False
        pred = self.classifier.predict(emb_normed.reshape(1, -1))[0]
        return pred == 0

    @lru_cache(maxsize=256)
    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def classify_intent(self, query: str) -> str:
        # If no LLM client, be conservative
        if self.client is None:
            return "other"
        system_prompt = (
            "You are a tactile QA assistant. Classify the question type into one of: "
            "property / comparison / judgement / other. Output only one word."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=10,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip().lower()
        except Exception:
            return "other"

    def build_messages(self, prompt, image_path, ref_paths=None, history=None, system_role=None):
        if history is None:
            history = []
        if not system_role:
            system_role = "You are a professional tactile perception expert."
        messages = [{"role": "system", "content": system_role}]
        for qa in history[-3:]:
            messages.append({"role": "user", "content": qa["question"]})
            messages.append({"role": "assistant", "content": qa["answer"]})
        base64_image = self.encode_image_to_base64(image_path)
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}},
        ]
        if ref_paths:
            for rp in ref_paths:
                if not os.path.exists(rp):
                    continue
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image_to_base64(rp)}", "detail": "low"}
                })
        messages.append({"role": "user", "content": user_content})
        return messages

    def answer(self, image_path, user_query, history=None, force_tactile_expert=False, use_professional_prompt=True):
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY and install openai.")

        # Text-only QA
        if not image_path or not os.path.exists(image_path):
            system_role = (
                "You are a professional tactile perception expert. Please answer the user's tactile question based only on your expertise."
                "Please answer concisely, in no more than 2–3 sentences."
            )
            messages = [{"role": "system", "content": system_role}]
            if history:
                for qa in history[-3:]:
                    messages.append({"role": "user", "content": qa["question"]})
                    messages.append({"role": "assistant", "content": qa["answer"]})
            messages.append({"role": "user", "content": user_query})
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=128,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()

        # With image: one ImageBind forward
        emb_raw = self.extract_raw_embedding(image_path)

        # Route: RGB vs Tactile
        if self.is_rgb(emb_raw) and not force_tactile_expert:
            system_role = "You are a tactile assistant that describes object tactile properties. Provide a concise response in no more than 2 sentences(30words)."
            prompt = user_query
            messages = self.build_messages(prompt, image_path, system_role=system_role, history=history)
        else:
            # ===== TACTILE BRANCH (tac→rgb labels for prompting) =====
            intent = self.classify_intent(user_query)
            topk = self.intent_topk.get(intent, 2)

            # Map query to rgb space with projector if available; else use raw
            if self.projector is not None:
                q_vec = self.apply_projector_to_vector(emb_raw)
            else:
                q_vec = emb_raw

            # Search RGB index; neighbors drive labels
            rgb_neighbors = self._search_ids(self.rgb_index, q_vec, topk)

            # Collect labels from rgb neighbor IDs
            ref_labels: List[str] = []
            ref_rgb_paths: List[str] = []
            for (rid, _) in rgb_neighbors:
                key = str(rid)
                label = self.idx2caption.get(key, "<unknown>") if self.idx2caption else "<unknown>"
                ref_labels.append(label)
                ref_rgb_paths.append(os.path.join(self.rgb_img_dir, f"image_{rid}_rgb.jpg"))

            captions_text = "; ".join([f"{i+1}. {c}" for i, c in enumerate(ref_labels)])
            if use_professional_prompt:
                intent_instruction_map = {
                    "property": ("Focus on touch-based attributes. Do not speculate visual characteristics like color, shape, or size."),
                    "comparison": ("Compare the tactile feeling of the target sample with the reference labels. Emphasize tactile attributes only."),
                    "judgement": ("Infer the possible material or physical condition of the object based on tactile characteristics. State uncertainty if needed."),
                    "other": ("Respond using only tactile reasoning. Use the reference tactile labels to support your response."),
                }
                task_instruction = intent_instruction_map.get(intent, intent_instruction_map["other"])
                prompt = (
                    f"You are given a tactile sensor image.\n"
                    f"The following tactile sample labels describe sensations from similar reference objects (from rgb-nearest samples):\n{captions_text}\n"
                    f"Use the tactile labels as supporting evidence. Focus solely on tactile perception.\n"
                    f"{task_instruction}\n"
                    f"User Question:{user_query}\n"
                )
                system_role = (
                    "You are a tactile perception expert. Answer only based on tactile features. "
                    "Provide a concise response (<=2 sentences, <=30 words)."
                )

            base64_image = self.encode_image_to_base64(image_path)
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}},
            ]
            for rgb_path in ref_rgb_paths:
                if os.path.exists(rgb_path):
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image_to_base64(rgb_path)}", "detail": "low"}
                    })

            messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_content},
            ]

        response = self.client.chat.completions.create(
            model=getattr(self, "ft_model_name", None) or "ft:gpt-4o-2024-08-06:personal:tactilemode:C3Y5BzQp",
            messages=messages,
            max_tokens=128,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Quick smoke test (requires OPENAI_API_KEY if calling answer())
    model = TouchQAModel(
        tactile_emb_dir="embeddings/embeddings_tac",
        rgb_emb_dir="embeddings/embeddings_rgb",
        projector_path="tac_projector_vit5p_best.pt",  # or '' to disable
        caption_csv="data/ssvtp/new_train.csv",
        tactile_img_dir="data/ssvtp/images_tac",
        rgb_img_dir="data/ssvtp/images_rgb",
        projector_sensor_id=None,
        min_cos=0.0,
        overfetch_mul=2,
        overfetch_bias=5,
        normalize_before_projector=False,
    )
    # arrays for eval
    tac_raw, tac_ids, rgb_emb, rgb_ids = model.get_arrays_for_eval()
    print("Loaded:", tac_raw.shape, len(tac_ids), rgb_emb.shape, len(rgb_ids))
