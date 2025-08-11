import os
import numpy as np
import torch
from tqdm import tqdm

# 导入最新 ImageBind 模型组件（确保安装官方最新版 imagebind）
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind.data import load_and_transform_vision_data

# 路径设置
image_dir = "./data/ssvtp/images_rgb"            # GelSight 图像目录
save_root = "./embeddings/embeddings_rgb"        # embedding 保存目录
os.makedirs(save_root, exist_ok=True)

# 加载 imagebind_huge 官方预训练模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()

# 提取 embedding
image_list = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
all_embeddings = []
all_indices = []

for img_name in tqdm(image_list, desc="Extracting embeddings"):
    # 假设文件名为 xxx_0123.jpg，提取 index = "0123"
    idx = img_name.split("_")[1].split(".")[0]
    image_path = os.path.join(image_dir, img_name)

    inputs = {
        ModalityType.VISION: load_and_transform_vision_data([image_path], device),
    }

    with torch.no_grad():
        embedding = model(inputs)[ModalityType.VISION]  # shape: (1, 1024)

    embedding_np = embedding.cpu().numpy()[0]
    all_embeddings.append(embedding_np)
    all_indices.append(idx)

    # 可选：保存单个 embedding 文件
    np.save(os.path.join(save_root, f"{idx}_embedding.npy"), embedding_np)

# 保存所有 embedding 和索引
np.save(os.path.join(save_root, "all_embeddings.npy"), np.stack(all_embeddings))
np.save(os.path.join(save_root, "all_indices.npy"), np.array(all_indices))
