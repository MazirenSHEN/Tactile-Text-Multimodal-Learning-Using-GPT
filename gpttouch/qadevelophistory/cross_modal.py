import torch
import numpy as np
import faiss
from imagebind.models.imagebind_model import imagebind_huge, ModalityType
from imagebind.data import load_and_transform_vision_data
from openai import OpenAI
import os
import base64

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CrossModalRetriever:
    def __init__(self, tactile_dir, rgb_dir, emb_dim=1024):
        # Tactile
        self.tac_emb = np.load(os.path.join(tactile_dir, "all_embeddings.npy"))
        self.tac_idx = np.load(os.path.join(tactile_dir, "all_indices.npy"))
        self.tac_emb_norm = self.tac_emb / np.linalg.norm(self.tac_emb, axis=1, keepdims=True)
        self.tac_index = faiss.IndexFlatIP(emb_dim)
        self.tac_index.add(self.tac_emb_norm.astype("float32"))
        # RGB
        self.rgb_emb = np.load(os.path.join(rgb_dir, "all_embeddings.npy"))
        self.rgb_idx = np.load(os.path.join(rgb_dir, "all_indices.npy"))
        self.rgb_emb_norm = self.rgb_emb / np.linalg.norm(self.rgb_emb, axis=1, keepdims=True)
        self.rgb_index = faiss.IndexFlatIP(emb_dim)
        self.rgb_index.add(self.rgb_emb_norm.astype("float32"))

    def retrieve(self, emb, modality="tactile", topk=3):
        # emb: 已归一化的 query 向量
        if modality == "tactile":
            D, I = self.tac_index.search(emb.reshape(1, -1).astype("float32"), topk)
            return [self.tac_idx[i] for i in I[0]], D[0]
        else:
            D, I = self.rgb_index.search(emb.reshape(1, -1).astype("float32"), topk)
            return [self.rgb_idx[i] for i in I[0]], D[0]

class CrossModalQAModel:
    def __init__(self, tactile_dir, rgb_dir, client=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = imagebind_huge(pretrained=True).to(self.device).eval()
        self.retriever = CrossModalRetriever(tactile_dir, rgb_dir)
        self.client = client  # OpenAI 客户端，如 OpenAI(api_key=...)

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def extract_embedding(self, image_path):
        vis = load_and_transform_vision_data([image_path], self.device)
        with torch.no_grad():
            emb = self.model({ModalityType.VISION: vis})[ModalityType.VISION]
        emb = emb.cpu().numpy()[0]
        emb = emb / np.linalg.norm(emb)
        return emb

    def cross_modal_retrieve(self, image_path, query_from="rgb", topk=3):
        emb = self.extract_embedding(image_path)
        if query_from == "rgb":
            indices, scores = self.retriever.retrieve(emb, modality="tactile", topk=topk)
            data_dir = "data/ssvtp/images_tac"
            # 生成 tactile 文件名，比如 image_0_tac.jpg
            file_paths = [os.path.join(data_dir, f"image_{idx}_tac.jpg") for idx in indices]
        else:
            indices, scores = self.retriever.retrieve(emb, modality="rgb", topk=topk)
            data_dir = "data/ssvtp/images_rgb"
            file_paths = [os.path.join(data_dir, f"image_{idx}_rgb.jpg") for idx in indices]
        return file_paths, scores

    def cross_modal_qa(self, image_path, user_query, query_from="rgb", topk=3):
        # 1. 跨模态检索得到topk目标图像
        file_paths, scores = self.cross_modal_retrieve(image_path, query_from, topk)
        # 2. base64 编码：原始输入图片 + 检索图片
        base64_input = self.encode_image_to_base64(image_path)
        base64_cross = [self.encode_image_to_base64(fp) for fp in file_paths]

        # 3. 构造多模态 prompt
        if query_from == "rgb":
            system_role = "You are an expert in material perception. Please describe the likely tactile sensation of the main object in the given RGB image. Reference tactile images are provided for context."
            user_content = [
                {"type": "text", "text": f"{user_query}\nThe following tactile samples are retrieved for reference:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_input}", "detail": "low"}}
            ]
            for b64 in base64_cross:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
        else:
            system_role = "You are an expert in visual perception. Please describe the likely visual appearance of the object in the given tactile image. Reference RGB images are provided for context."
            user_content = [
                {"type": "text", "text": f"{user_query}\nThe following RGB samples are retrieved for reference:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_input}", "detail": "low"}}
            ]
            for b64 in base64_cross:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})

        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_content}
        ]

        # 4. GPT-4o 推理
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=256,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()


# ====== 调用示例 ======
if __name__ == "__main__":
    openai_client = OpenAI(api_key="sk-proj-")
    qa_model = CrossModalQAModel("embeddings/embeddings_tac", "embeddings/embeddings_rgb", openai_client)
    reply = qa_model.cross_modal_qa("data/test_tac_1.jpg", "Describe the tactile feel.", query_from="tac")
    print(reply)