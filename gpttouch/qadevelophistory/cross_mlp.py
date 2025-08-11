import torch
import torch.nn as nn
import numpy as np
import faiss
from imagebind.models.imagebind_model import imagebind_huge, ModalityType
from imagebind.data import load_and_transform_vision_data
from openai import OpenAI
import os
import base64

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === 投影网络 ===
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

# === 跨模态检索器 ===
class CrossModalRetriever:
    def __init__(self, tactile_dir, rgb_dir, emb_dim=1024):
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
        if modality == "tactile":
            D, I = self.tac_index.search(emb.reshape(1, -1).astype("float32"), topk)
            return [self.tac_idx[i] for i in I[0]], D[0]
        else:
            D, I = self.rgb_index.search(emb.reshape(1, -1).astype("float32"), topk)
            return [self.rgb_idx[i] for i in I[0]], D[0]

# === 跨模态问答主模型 ===
class CrossModalQAModel:
    def __init__(self, tactile_dir, rgb_dir, client=None, projector_path="tac_projector.pt", emb_dim=1024):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = imagebind_huge(pretrained=True).to(self.device).eval()
        self.retriever = CrossModalRetriever(tactile_dir, rgb_dir, emb_dim=emb_dim)
        self.client = client  # OpenAI 客户端

        # 加载tactile->rgb投影对齐网络
        self.projector = TactileProjector(emb_dim=emb_dim).to(self.device)
        self.projector.load_state_dict(torch.load(projector_path, map_location=self.device))
        self.projector.eval()

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def extract_embedding(self, image_path, use_projector=False):
        vis = load_and_transform_vision_data([image_path], self.device)
        with torch.no_grad():
            emb = self.model({ModalityType.VISION: vis})[ModalityType.VISION]
        emb = emb.cpu().numpy()[0]
        emb = emb / np.linalg.norm(emb)
        if use_projector:
            emb_t = torch.from_numpy(emb).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.projector(emb_t).cpu().numpy()[0]
            emb = emb / np.linalg.norm(emb)
        return emb

    def cross_modal_retrieve(self, image_path, query_from="rgb", topk=3):
        if query_from == "rgb":
            emb = self.extract_embedding(image_path, use_projector=False)
            indices, scores = self.retriever.retrieve(emb, modality="tactile", topk=topk)
            data_dir = "data/ssvtp/images_tac"
            file_paths = [os.path.join(data_dir, f"image_{idx}_tac.jpg") for idx in indices]
        else:
            emb = self.extract_embedding(image_path, use_projector=True)  # tactile->rgb必须用对齐后的
            indices, scores = self.retriever.retrieve(emb, modality="rgb", topk=topk)
            data_dir = "data/ssvtp/images_rgb"
            file_paths = [os.path.join(data_dir, f"image_{idx}_rgb.jpg") for idx in indices]
        return file_paths, scores



    def cross_modal_qa(self, image_path, user_query, query_from="rgb", topk=3):
        file_paths, scores = self.cross_modal_retrieve(image_path, query_from, topk)
        base64_input = self.encode_image_to_base64(image_path)
        base64_cross = [self.encode_image_to_base64(fp) for fp in file_paths]

        if query_from == "rgb":
            # （原逻辑不变）
            system_role = "You are an expert in material perception. Please describe the likely tactile sensation of the main object in the given RGB image. Reference tactile images are provided for context."
            user_content = [
                {"type": "text", "text": f"{user_query}\nThe following tactile samples are retrieved for reference:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_input}", "detail": "low"}}
            ]
            for b64 in base64_cross:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
        else:
            # tactile 仅做触感描述
            system_role = (
                "You are a professional tactile perception expert. "
                "Given tactile sensor images and relevant tactile reference samples, "
                "please provide a detailed, objective description of the object's surface texture, material properties, and tactile sensations. "
                "Do NOT discuss the object's visual appearance, color, or shape. "
                "If some attributes cannot be inferred, make the best reasonable scientific guess based on tactile cues and references."
            )
            user_content = [
                {
                    "type": "text",
                    "text": (
                        f"{user_query}\n"
                        f"The first image is the target tactile sample. "
                        f"The following are {len(base64_cross)} reference tactile images selected for similarity. "
                        f"Please describe, with as much technical detail as possible, the object's tactile sensation, "
                        f"including but not limited to roughness, smoothness, hardness, elasticity, stickiness, temperature, and possible material type. "
                        f"If there is uncertainty, state it clearly."
                    )
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_input}", "detail": "low"}}
            ]
            for b64 in base64_cross:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})

        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_content}
        ]

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

    qa_model = CrossModalQAModel(
        tactile_dir="embeddings/embeddings_tac",
        rgb_dir="embeddings/embeddings_rgb",
        client=openai_client,
        projector_path="tac_projector.pt",   # 修改为你的模型路径
        emb_dim=1024
    )
    # tactile->RGB问答
    reply = qa_model.cross_modal_qa("data/test_tac_1.jpg", "Describe the likely visual appearance.", query_from="tac", topk=3)
    print(reply)
    # RGB->tactile问答（原始流程，无需投影）
    # reply2 = qa_model.cross_modal_qa("data/test_rgb_1.jpg", "Describe the tactile feel.", query_from="rgb", topk=3)
    # print(reply2)
