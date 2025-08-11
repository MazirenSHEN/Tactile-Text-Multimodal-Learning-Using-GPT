import torch
import numpy as np
import joblib
import os
import faiss
import pandas as pd
import random
from openai import OpenAI
from imagebind.models.imagebind_model import imagebind_huge, ModalityType
from imagebind.data import load_and_transform_vision_data

class TouchQAModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = imagebind_huge(pretrained=True).to(self.device).eval()
        self.classifier = joblib.load("../tactile_rgb_classifier.pkl")
        self.client = OpenAI(api_key=os.getenv("sk-proj-"))  # 建议用环境变量
        # faiss检索与描述
        self.emb_root = "embeddings/embeddings_tac"
        self.emb_dim = 1024
        self.top_k = 1
        self.all_emb = np.load(os.path.join(self.emb_root, "all_embeddings.npy"))
        self.all_idx = np.load(os.path.join(self.emb_root, "all_indices.npy"))
        self.emb_norm = self.all_emb / np.linalg.norm(self.all_emb, axis=1, keepdims=True)
        self.index = faiss.IndexFlatIP(self.emb_dim)
        self.index.add(self.emb_norm.astype("float32"))
        self.df = pd.read_csv("../data/ssvtp/new_train.csv", dtype={"index": str})
        with open("../data/ssvtp/text_prefix.txt", encoding="utf-8") as f:
            self.prefixes = [line.strip() for line in f if line.strip()]

    # 图像embedding提取
    def extract_embedding(self, image_path):
        vis = load_and_transform_vision_data([image_path], self.device)
        with torch.no_grad():
            emb = self.model({ModalityType.VISION: vis})[ModalityType.VISION]
        return emb.cpu().numpy()

    # 检测RGB or Tactile
    def is_rgb(self, emb):
        pred = self.classifier.predict(emb)[0]
        return pred == 0

    # FAISS检索top-k描述
    def retrieve_captions(self, emb_vec, k=None):
        if k is None:
            k = self.top_k
        emb_vec = emb_vec[0] / np.linalg.norm(emb_vec[0])
        D, I = self.index.search(emb_vec.reshape(1, -1).astype("float32"), k)
        captions = []
        for pos in I[0]:
            idx = self.all_idx[pos]
            row = self.df[self.df["index"] == idx]
            captions.append(row.iloc[0]["caption"] if not row.empty else "<unknown>")
        return captions

    # 简单自适应Prompt（可扩展为MLP/embedding聚类后风格切换）
    def adaptive_prompt(self, user_query, captions, style='auto'):
        # style可选: auto, professional, simple, comparison等
        # 这里先用auto, 可根据captions内容动态调整
        prefix = random.choice(self.prefixes)
        prompt = f"{prefix} {user_query}, Related to the following tactile cases:{'; '.join(captions)}. Please integrate all these pieces of information to generate a more accurate and human-like professional description that closely reflects real human perception."
        return prompt

    # 多轮上下文历史拼接
    def build_multi_turn_messages(self, prompt, history, role="user"):
        messages = [{"role": "system", "content": "You are a professional expert in tactile perception, skilled at answering questions related to touch in either simple or technical terms."}]
        for qa in history[-3:]:  # 最多引用3轮历史
            messages.append({"role": "user", "content": qa["question"]})
            messages.append({"role": "assistant", "content": qa["answer"]})
        messages.append({"role": role, "content": prompt})
        return messages

    # 核心主流程
    def answer(self, image_path, user_query, history=None):
        if history is None:
            history = []
        emb = self.extract_embedding(image_path)
        if self.is_rgb(emb):
            # RGB分流（可拓展为视觉描述，略）
            prompt = f"{user_query}（This image is of RGB type. Please use visual information for reasoning.）"
            messages = self.build_multi_turn_messages(prompt, history)
        else:
            captions = self.retrieve_captions(emb)
            prompt = self.adaptive_prompt(user_query, captions, style='auto')
            messages = self.build_multi_turn_messages(prompt, history)
        # GPT-4o调用
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=256,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
