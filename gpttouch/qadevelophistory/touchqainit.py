import torch
import numpy as np
import joblib
import os
import faiss
import pandas as pd
import random
import base64
from openai import OpenAI
from imagebind.models.imagebind_model import imagebind_huge, ModalityType
from imagebind.data import load_and_transform_vision_data

class TouchQAModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = imagebind_huge(pretrained=True).to(self.device).eval()
        self.classifier = joblib.load("../tactile_rgb_classifier.pkl")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.emb_root = "embeddings/embeddings_tac"
        self.emb_dim = 1024
        self.all_emb = np.load(os.path.join(self.emb_root, "all_embeddings.npy"))
        self.all_idx = np.load(os.path.join(self.emb_root, "all_indices.npy"))
        self.emb_norm = self.all_emb / np.linalg.norm(self.all_emb, axis=1, keepdims=True)
        self.index = faiss.IndexFlatIP(self.emb_dim)
        self.index.add(self.emb_norm.astype("float32"))

        self.df = pd.read_csv("../data/ssvtp/new_train.csv", dtype={"index": str})
        with open("../data/ssvtp/text_prefix.txt", encoding="utf-8") as f:
            self.prefixes = [line.strip() for line in f if line.strip()]

        self.intent_prompt_templates = {
            "property": "Please describe the tactile perception of the image based on the following cases: {captions}.",
            "comparison": "Compare the tactile features in the image with the following cases: {captions}.",
            "judgement": "Given the tactile cues: {captions}, infer the likely material or physical state of the object.",
            "repair": "Based on tactile characteristics: {captions}, suggest possible reasons for the abnormal touch and propose a solution.",
            "other": "Based on the tactile reference cases: {captions}, answer the following user question."
        }
        # 根据意图自动决定检索参考caption数量
        self.intent_topk = {
            "property": 1,
            "comparison": 2,
            "judgement": 2,
            "repair": 2,
            "other": 1
        }

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def extract_embedding(self, image_path):
        vis = load_and_transform_vision_data([image_path], self.device)
        with torch.no_grad():
            emb = self.model({ModalityType.VISION: vis})[ModalityType.VISION]
        return emb.cpu().numpy()

    def is_rgb(self, emb):
        pred = self.classifier.predict(emb)[0]
        return pred == 0

    def retrieve_captions(self, emb_vec, k=1):
        emb_vec = emb_vec[0] / np.linalg.norm(emb_vec[0])
        D, I = self.index.search(emb_vec.reshape(1, -1).astype("float32"), k)
        captions = []
        for pos in I[0]:
            idx = self.all_idx[pos]
            row = self.df[self.df["index"] == idx]
            captions.append(row.iloc[0]["caption"] if not row.empty else "<unknown>")
        return captions

    def classify_intent(self, query):
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

    def adaptive_prompt(self, user_query, captions, intent=None):
        if intent is None:
            intent = self.classify_intent(user_query)
        template = self.intent_prompt_templates.get(intent, self.intent_prompt_templates["other"])
        caption_text = "; ".join(captions)
        prompt = template.format(captions=caption_text)
        prompt += f"\nUser question: {user_query}"
        return prompt

    def build_multi_turn_messages(self, prompt, image_path, history=None, system_role="You are a professional expert in tactile perception, skilled at answering questions related to touch in either simple or technical terms."):
        if history is None:
            history = []
        messages = [{"role": "system", "content": system_role}]
        for qa in history[-3:]:
            messages.append({"role": "user", "content": qa["question"]})
            messages.append({"role": "assistant", "content": qa["answer"]})
        base64_image = self.encode_image_to_base64(image_path)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                }}
            ]
        })
        return messages

    def answer(self, image_path, user_query, history=None):
        emb = self.extract_embedding(image_path)
        if self.is_rgb(emb):
            # RGB图片
            system_role = "You are a vision assistant that describes object properties."
            prompt = user_query
        else:
            # Tactile图片，意图识别、动态topk
            intent = self.classify_intent(user_query)
            topk = self.intent_topk.get(intent, 1)
            captions = self.retrieve_captions(emb, k=topk)
            prompt = self.adaptive_prompt(user_query, captions, intent=intent)
            system_role = "You are a tactile expert. Please answer based on the touch and surface characteristics of the tactile image."
        messages = self.build_multi_turn_messages(prompt, image_path, history, system_role)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=256,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
