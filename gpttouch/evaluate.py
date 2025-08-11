from sentence_transformers import SentenceTransformer, util
import pandas as pd

# 读取结果文件
df = pd.read_csv("result2.csv")
model = SentenceTransformer('all-mpnet-base-v2')  # 可替换更强模型

# 初始化相似度列表
sim_to_label = []
sim_to_reference = []
sim_label_vs_reference = []

for _, row in df.iterrows():
    answer = row.get('model_answer', '')
    label = row.get('label', '')
    references = row['reference_labels'].split('|') if pd.notna(row['reference_labels']) else []
    reference_text = ' '.join(references)

    # 判断缺失或空值
    if not answer or not label or not reference_text.strip():
        sim_to_label.append(None)
        sim_to_reference.append(None)
        sim_label_vs_reference.append(None)
        continue

    # 编码
    emb_answer = model.encode(answer, convert_to_tensor=True)
    emb_label = model.encode(label, convert_to_tensor=True)
    emb_reference = model.encode(reference_text, convert_to_tensor=True)

    # 计算三种相似度
    sim_label = util.cos_sim(emb_answer, emb_label).item()
    sim_ref = util.cos_sim(emb_answer, emb_reference).item()
    sim_label_ref = util.cos_sim(emb_label, emb_reference).item()

    sim_to_label.append(sim_label)
    sim_to_reference.append(sim_ref)
    sim_label_vs_reference.append(sim_label_ref)

# 写入结果
df['sim_to_label'] = sim_to_label
df['sim_to_reference'] = sim_to_reference
df['sim_label_vs_reference_labels'] = sim_label_vs_reference

df.to_csv("similarity_metrics3.csv", index=False, encoding="utf-8-sig")
# 统计均值（跳过None/NaN）
sim_to_label_mean = df['sim_to_label'].dropna().mean()
sim_to_reference_mean = df['sim_to_reference'].dropna().mean()
sim_label_vs_reference_mean = df['sim_label_vs_reference_labels'].dropna().mean()

# 打印
print(f"sim_to_label mean: {sim_to_label_mean:.4f}")
print(f"sim_to_reference mean: {sim_to_reference_mean:.4f}")
print(f"sim_label_vs_reference_labels mean: {sim_label_vs_reference_mean:.4f}")

# 保存到txt
with open("similarity_metrics3_mean.txt", "w", encoding="utf-8") as f:
    f.write(f"sim_to_label mean: {sim_to_label_mean:.4f}\n")
    f.write(f"sim_to_reference mean: {sim_to_reference_mean:.4f}\n")
    f.write(f"sim_label_vs_reference_labels mean: {sim_label_vs_reference_mean:.4f}\n")

print("✅ 已完成三方相似度评估（模型输出 vs label / reference，以及 label vs reference）。")
