import numpy as np
import pandas as pd

csv_path = "../data/ssvtp/new_train.csv"   # 你的 CSV 文件路径
emb_path = "../embeddings/embeddings_tac/all_embeddings.npy"    # embedding 文件路径
idx_path = "../embeddings/embeddings_tac/all_indices.npy"       # index 文件路径

df = pd.read_csv(csv_path)
embeddings = np.load(emb_path)
indices = np.load(idx_path)

csv_indices = df["index"].astype(str).values
embedding_indices = np.array([str(i) for i in indices])

if np.array_equal(csv_indices, embedding_indices):
    print("✅ embedding 文件和 CSV 顺序完全一致！")
else:
    print("❗ 顺序不一致，开始自动对齐 ...")
    # 获取 CSV 每行 index 在 embedding 的顺序
    reorder = [list(embedding_indices).index(i) for i in csv_indices]
    embeddings = embeddings[reorder]
    # 保存为新的对齐好的 embedding 文件
    np.save("all_embeddings_aligned.npy", embeddings)
    print("✅ 已生成 all_embeddings_aligned.npy，顺序已与 CSV 一致！")
