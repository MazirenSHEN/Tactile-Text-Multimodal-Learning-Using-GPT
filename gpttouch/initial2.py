import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import faiss
import pandas as pd
import base64
from openai import OpenAI
from imagebind.models.imagebind_model import imagebind_huge, ModalityType
from imagebind.data import load_and_transform_vision_data

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

class TouchQAModel:
    def __init__(
        self,
        tactile_emb_dir,
        rgb_emb_dir,
        projector_path=None,
        caption_csv=None,
        emb_dim=1024,
        tactile_img_dir="data/ssvtp/images_tac",
        rgb_img_dir="data/ssvtp/images_rgb"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = imagebind_huge(pretrained=True).to(self.device).eval()
        self.classifier = joblib.load("tactile_rgb_classifier.pkl")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Tactile embedding检索库
        self.tac_emb = np.load(os.path.join(tactile_emb_dir, "all_embeddings.npy"))
        self.tac_idx = np.load(os.path.join(tactile_emb_dir, "all_indices.npy"))
        self.tac_emb_norm = self.tac_emb / np.linalg.norm(self.tac_emb, axis=1, keepdims=True)
        self.tac_index = faiss.IndexFlatIP(emb_dim)
        self.tac_index.add(self.tac_emb_norm.astype("float32"))
        # RGB embedding检索库
        self.rgb_emb = np.load(os.path.join(rgb_emb_dir, "all_embeddings.npy"))
        self.rgb_idx = np.load(os.path.join(rgb_emb_dir, "all_indices.npy"))
        self.rgb_emb_norm = self.rgb_emb / np.linalg.norm(self.rgb_emb, axis=1, keepdims=True)
        self.rgb_index = faiss.IndexFlatIP(emb_dim)
        self.rgb_index.add(self.rgb_emb_norm.astype("float32"))

        # 投影网络（可选）
        self.projector = None
        if projector_path:
            self.projector = TactileProjector(emb_dim).to(self.device)
            self.projector.load_state_dict(torch.load(projector_path, map_location=self.device))
            self.projector.eval()

        # caption 检索
        self.df = pd.read_csv(caption_csv, dtype={"index": str}) if caption_csv else None

        # 其它参数
        self.emb_dim = emb_dim
        self.tactile_img_dir = tactile_img_dir
        self.rgb_img_dir = rgb_img_dir

        # 意图分流和动态topk
        self.intent_prompt_templates = {
            "property": "Please describe the tactile perception of the object based on the following reference sample labels: {captions}.",
            "comparison": "Compare the tactile features of the object with these reference sample labels: {captions}.",
            "judgement": "Given these tactile cues: {captions}, infer the likely material or physical state of the object.",
            # "repair": "Based on these tactile characteristics: {captions}, suggest possible reasons for the abnormal touch and propose a solution.",
            "other": "Based on the tactile reference labels: {captions}, answer the following user question."
        }
        self.intent_topk = {
            "property": 1,
            "comparison": 2,
            "judgement": 2,
            # "repair": 2,
            "other": 2
        }

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def extract_embedding(self, image_path, use_projector=False):
        vis = load_and_transform_vision_data([image_path], self.device)
        with torch.no_grad():
            emb = self.model({ModalityType.VISION: vis})[ModalityType.VISION]
        emb = emb.cpu().numpy()[0]
        emb = emb / np.linalg.norm(emb)
        if use_projector and self.projector is not None:
            emb_t = torch.from_numpy(emb).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.projector(emb_t).cpu().numpy()[0]
            emb = emb / np.linalg.norm(emb)
        return emb

    def is_rgb(self, emb):
        pred = self.classifier.predict(emb.reshape(1, -1))[0]
        return pred == 0

    def classify_intent(self, query):
        # 用gpt意图识别
        system_prompt = "你是一个触觉问答助手，请根据问题类型分类为：property / comparison / judgement / repair / other。只输出一个词。"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=10,
            temperature=0.0
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
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}}
        ]
        if ref_paths:
            for rp in ref_paths:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image_to_base64(rp)}", "detail": "low"}})
        messages.append({"role": "user", "content": user_content})
        return messages

    def answer(
        self,
        image_path,
        user_query,
        history=None,
        force_tactile_expert=False,
        use_professional_prompt=True
    ):
        # 1. 处理“仅文本问答”场景
        if not image_path or not os.path.exists(image_path):
            # 构造system_role
            system_role = (
                "You are a professional tactile perception expert. Please answer the user's tactile question based only on your expertise."
                "Please answer concisely, in no more than 2–3 sentences."
            )
            # 构造消息体
            #messages = [{"role": "system", "content": system_role}]
            messages = [
                {"role": "system", "content": system_role}
            ]

            if history:
                for qa in history[-3:]:
                    messages.append({"role": "user", "content": qa["question"]})
                    messages.append({"role": "assistant", "content": qa["answer"]})
            messages.append({"role": "user", "content": user_query})
            # 调用GPT-4o
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=128,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()

        emb = self.extract_embedding(image_path)
        if self.is_rgb(emb) and not force_tactile_expert:
            # RGB分流
            system_role = "You are a vision assistant that describes object properties. Provide a concise response in no more than 2 sentences(30words)."
            prompt = user_query
            messages = self.build_messages(prompt, image_path, system_role=system_role, history=history)
        else:
            # tactile分流
            use_proj = self.projector is not None
            emb_tac = self.extract_embedding(image_path, use_projector=use_proj)
            # === 意图识别分流 ===
            intent = self.classify_intent(user_query)
            topk = self.intent_topk.get(intent, 2)
            emb_norm = emb_tac / np.linalg.norm(emb_tac)
            D, I = self.tac_index.search(emb_norm.reshape(1, -1).astype("float32"), topk)
            ref_labels, ref_rgb_paths = [], []
            for pos in I[0]:
                idx = self.tac_idx[pos]
                # tactile label
                if self.df is not None:
                    row = self.df[self.df["index"] == idx]
                    label = row.iloc[0]["caption"] if not row.empty else "<unknown>"
                else:
                    label = "<unknown>"
                ref_labels.append(label)
                # 对齐rgb
                rgb_img_path = os.path.join(self.rgb_img_dir, f"image_{idx}_rgb.jpg")
                ref_rgb_paths.append(rgb_img_path)
            # === 自适应prompt模式：intent模板+label+rgb辅助 ===
            captions_text = "; ".join([f"{i+1}. {c}" for i, c in enumerate(ref_labels)])
            if use_professional_prompt:
                intent_instruction_map = {
                    "property": (
                        "Focus on touch-based attributes."
                        "Do not speculate visual characteristics like color, shape, or size."
                    ),
                    "comparison": (
                        "Compare the tactile feeling of the target sample with the reference labels. "
                        "Emphasize similarities and differences in tactile attributes only, not visual traits."
                    ),
                    "judgement": (
                        "Infer the possible material or physical condition of the object based on tactile characteristics."
                        "using the reference labels as guidance. Be cautious and clearly state uncertainty if needed."
                    ),
                    # "repair": (
                    #     "Evaluate whether the tactile sensation of the target object is abnormal or defective. "
                    # ),
                    "other": (
                        "Respond using only tactile reasoning. Do not include visual speculation. "
                        "Use the reference tactile labels to support your response."
                    )
                }

                task_instruction = intent_instruction_map.get(intent, intent_instruction_map["other"])

                prompt = (
                    f"You are given a tactile sensor image representing the physical sensation of contact with an object.\n"
                    f"The following tactile sample labels describe sensations from similar reference objects:\n{captions_text}\n"
                    f"Use the tactile labels as supporting evidence. \n"
                    f"Simply refer to the RGB image content,but Focus solely on tactile perception and the reference lables.\n"
                    f"{task_instruction}\n"
                    f"User Question:{user_query}\n"
                )
                system_role = (
                    "You are a tactile perception expert. "
                    "Answer only based on tactile features. Never describe or infer any visual property."
                    "Provide a concise and efficient response that closely aligns with the provided reference content, in no more than 2 sentences(30words)."

                )

            # === 构造消息 ===
            base64_image = self.encode_image_to_base64(image_path)
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}}
            ]
            for rgb_path in ref_rgb_paths:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image_to_base64(rgb_path)}", "detail": "low"}})
            messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_content}
            ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=128,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "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"
    qa_model = TouchQAModel(
        tactile_emb_dir="embeddings/embeddings_tac",
        rgb_emb_dir="embeddings/embeddings_rgb",
        projector_path="tac_projector.pt",  # 如无MLP可传None
        caption_csv="data/ssvtp/new_train.csv",
        tactile_img_dir="data/ssvtp/images_tac",
        rgb_img_dir="data/ssvtp/images_rgb"
    )
    reply = qa_model.answer(
        "",
        "毛巾的触感是怎么样的"
    )
    print(reply)
