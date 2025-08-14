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
from openai import OpenAI
from imagebind.models.imagebind_model import imagebind_huge, ModalityType
from imagebind.data import load_and_transform_vision_data

# =============================
# ViTProjector (same structure as the training script)
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

        self.gamma = nn.Parameter(torch.ones(emb_dim) * layerscale_init) if residual else None

        if learnable_tau:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau)))
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
    Supports either {'state_dict', 'config'} dicts or plain state_dict files.
    """
    ckpt = torch.load(projector_path, map_location=device)

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        cfg = ckpt.get('config', {})
        model = ViTProjector(
            emb_dim=cfg.get('emb_dim', default_emb_dim),
            num_tokens=cfg.get('num_tokens', 16),
            dim=cfg.get('dim', 256),
            depth=cfg.get('depth', 4),
            heads=cfg.get('heads', 8),
            num_sensors=cfg.get('num_sensors', 0),
            sensor_token_len=cfg.get('sensor_token_len', 5),
            learnable_tau=cfg.get('learnable_tau', True),
            init_tau=cfg.get('init_tau', 0.07),
            residual=cfg.get('residual', True),
            layerscale_init=cfg.get('layerscale_init', 1e-3),
        )
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        if missing or unexpected:
            print(f"[load_vit_projector] missing={missing}, unexpected={unexpected}")
        return model.to(device).eval()

    # fallback: assume ckpt itself is a state_dict
    model = ViTProjector(emb_dim=default_emb_dim)
    model.load_state_dict(ckpt, strict=False)
    return model.to(device).eval()


# =============================
# TouchQA model using ViT projector (optimized retrieval)
# =============================
class TouchQAModel:
    def __init__(
        self,
        tactile_emb_dir,
        rgb_emb_dir,
        projector_path,
        caption_csv,
        emb_dim: int = 1024,
        tactile_img_dir: str = "data/ssvtp/images_tac",
        rgb_img_dir: str = "data/ssvtp/images_rgb",
        projector_sensor_id: int | None = None,
        # retrieval config
        min_cos: float = 0.15,
        overfetch_mul: int = 2,
        overfetch_bias: int = 5,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = imagebind_huge(pretrained=True).to(self.device).eval()
        self.classifier = joblib.load("tactile_rgb_classifier.pkl")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.emb_dim = emb_dim
        self.min_cos = float(min_cos)
        self.overfetch_mul = int(overfetch_mul)
        self.overfetch_bias = int(overfetch_bias)

        # ----- Load embeddings (mmap, float32, L2-normalized) ------
        def _f32c(x: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(x.astype("float32", copy=False))

        # tactile db
        self.tac_emb = _f32c(np.load(os.path.join(tactile_emb_dir, "all_embeddings.npy"), mmap_mode='r'))
        self.tac_idx = np.load(os.path.join(tactile_emb_dir, "all_indices.npy"))
        assert self.tac_emb.shape[1] == emb_dim, f"tactile emb dim {self.tac_emb.shape[1]} != {emb_dim}"
        faiss.normalize_L2(self.tac_emb)
        self.tac_index = self._build_idmap_index(self.tac_emb, self.tac_idx)

        # rgb db
        self.rgb_emb = _f32c(np.load(os.path.join(rgb_emb_dir, "all_embeddings.npy"), mmap_mode='r'))
        self.rgb_idx = np.load(os.path.join(rgb_emb_dir, "all_indices.npy"))
        assert self.rgb_emb.shape[1] == emb_dim, f"rgb emb dim {self.rgb_emb.shape[1]} != {emb_dim}"
        faiss.normalize_L2(self.rgb_emb)
        self.rgb_index = self._build_idmap_index(self.rgb_emb, self.rgb_idx)

        # ----- Projector (optional) -----
        self.projector: Optional[nn.Module] = None
        self.projector_sensor_id = projector_sensor_id
        if projector_path:
            self.projector = load_vit_projector(projector_path, device=self.device, default_emb_dim=emb_dim)
            self.projector.eval()

        # Captions table: build O(1) map
        self.df = pd.read_csv(caption_csv, dtype={"index": str}) if caption_csv else None
        self.idx2caption = dict(zip(self.df["index"], self.df["caption"])) if self.df is not None else None

        # Image roots
        self.tactile_img_dir = tactile_img_dir
        self.rgb_img_dir = rgb_img_dir

        # Intent templates and dynamic top-k
        self.intent_prompt_templates = {
            "property": "Please describe the tactile perception of the object based on the following reference sample labels: {captions}.",
            "comparison": "Compare the tactile features of the object with these reference sample labels: {captions}.",
            "judgement": "Given these tactile cues: {captions}, infer the likely material or physical state of the object.",
            "other": "Based on the tactile reference labels: {captions}, answer the following user question.",
        }
        self.intent_topk = {"property": 1, "comparison": 2, "judgement": 2, "other": 2}

    # ---------- Index utils ----------
    def _build_idmap_index(self, xb: np.ndarray, ids: np.ndarray) -> faiss.Index:
        """Build IndexFlatIP + IDMap2, add_with_ids using real IDs.
        xb must be float32 L2-normalized, contiguous. ids will be cast to int64.
        """
        base = faiss.IndexFlatIP(self.emb_dim)
        index = faiss.IndexIDMap2(base)
        index.add_with_ids(xb, ids.astype('int64'))
        return index

    # ---------- Embedding utils ----------
    @torch.no_grad()
    def extract_raw_embedding(self, image_path: str) -> np.ndarray:
        """One ImageBind forward to get a normalized float32 vector."""
        vis = load_and_transform_vision_data([image_path], self.device)
        emb = self.model({ModalityType.VISION: vis})[ModalityType.VISION]
        emb = emb.detach().to("cpu").numpy()[0].astype("float32")
        # L2 normalize in-place via faiss (fast and stable)
        emb = np.ascontiguousarray(emb)
        faiss.normalize_L2(emb.reshape(1, -1))
        return emb

    @torch.no_grad()
    def apply_projector_to_vector(self, emb_raw: np.ndarray) -> np.ndarray:
        assert self.projector is not None
        x = torch.from_numpy(emb_raw).float().unsqueeze(0).to(self.device)
        if self.projector_sensor_id is not None:
            sid = torch.tensor([self.projector_sensor_id], device=self.device)
            y = self.projector(x, sensor_ids=sid)
        else:
            y = self.projector(x)
        y = y.detach().to("cpu").numpy()[0].astype("float32")
        y = np.ascontiguousarray(y)
        faiss.normalize_L2(y.reshape(1, -1))
        return y

    def is_rgb(self, emb_normed: np.ndarray) -> bool:
        # Classifier expects 1D feature; emb_normed is already L2 normalized float32
        pred = self.classifier.predict(emb_normed.reshape(1, -1))[0]
        return pred == 0

    # ---------- Retrieval helpers ----------
    def _search_ids(self, index: faiss.Index, q: np.ndarray, topk: int) -> List[Tuple[int, float]]:
        """Return list of (real_id, score). q must be 1D float32.
        Overfetch + min_cos filtering to reduce noise.
        """
        q = np.ascontiguousarray(q.reshape(1, -1).astype("float32", copy=False))
        faiss.normalize_L2(q)
        k = max(topk * self.overfetch_mul, topk + self.overfetch_bias)
        D, I = index.search(q, k)
        res: List[Tuple[int, float]] = []
        used = set()
        for id_, s in zip(I[0].tolist(), D[0].tolist()):
            if id_ == -1 or s < self.min_cos or id_ in used:
                continue
            used.add(id_)
            res.append((id_, float(s)))
            if len(res) >= topk:
                break
        return res

    # ---------- Messaging helpers ----------
    @lru_cache(maxsize=256)
    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def classify_intent(self, query: str) -> str:
        system_prompt = "你是一个触觉问答助手，请根据问题类型分类为：property / comparison / judgement / other。只输出一个词。"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=10,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip().lower()

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
                try:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image_to_base64(rp)}", "detail": "low"}
                    })
                except FileNotFoundError:
                    # 如果参考图不存在，静默跳过
                    continue
        messages.append({"role": "user", "content": user_content})
        return messages

    # ---------- Main QA ----------
    def answer(self, image_path, user_query, history=None, force_tactile_expert=False, use_professional_prompt=True):
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
            # RGB branch
            system_role = "You are a tactile assistant that describes object tactile properties. Provide a concise response in no more than 2 sentences(30words)."
            prompt = user_query
            messages = self.build_messages(prompt, image_path, system_role=system_role, history=history)
        else:
            # Tactile branch
            intent = self.classify_intent(user_query)
            topk = self.intent_topk.get(intent, 2)

            # --- tac→tac for captions (NO projector) ---
            tac_neighbors = self._search_ids(self.tac_index, emb_raw, topk)

            ref_labels: List[str] = []
            for (tid, _) in tac_neighbors:
                key = str(tid)
                label = self.idx2caption.get(key, "<unknown>") if self.idx2caption else "<unknown>"
                ref_labels.append(label)

            # --- tac→rgb for reference RGB (WITH projector if available) ---
            ref_rgb_paths: List[str] = []
            if self.projector is not None:
                emb_proj = self.apply_projector_to_vector(emb_raw)
                rgb_neighbors = self._search_ids(self.rgb_index, emb_proj, topk)
                for (rid, _) in rgb_neighbors:
                    ref_rgb_paths.append(os.path.join(self.rgb_img_dir, f"image_{rid}_rgb.jpg"))
            else:
                # Fallback: use paired RGB of tac neighbors
                for (tid, _) in tac_neighbors:
                    ref_rgb_paths.append(os.path.join(self.rgb_img_dir, f"image_{tid}_rgb.jpg"))

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
                    f"The following tactile sample labels describe sensations from similar reference objects:\n{captions_text}\n"
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
                try:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image_to_base64(rgb_path)}", "detail": "low"}
                    })
                except FileNotFoundError:
                    continue

            messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_content},
            ]

        response = self.client.chat.completions.create(
            model="ft:gpt-4o-2024-08-06:personal:tactilemode:C3Y5BzQp",
            messages=messages,
            max_tokens=128,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Expect OPENAI_API_KEY in environment
    qa_model = TouchQAModel(
        tactile_emb_dir="embeddings/embeddings_tac",
        rgb_emb_dir="embeddings/embeddings_rgb",
        projector_path="tac_projector_vit5_best.pt",  # point to your trained ViT checkpoint, or '' to disable
        caption_csv="data/ssvtp/new_train.csv",
        tactile_img_dir="data/ssvtp/images_tac",
        rgb_img_dir="data/ssvtp/images_rgb",
        projector_sensor_id=None,  # e.g., 0 if trained with per-sensor tokens; None otherwise
        min_cos=0.15,
        overfetch_mul=2,
        overfetch_bias=5,
    )
