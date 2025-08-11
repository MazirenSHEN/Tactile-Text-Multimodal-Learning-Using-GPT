# evaluation_full.py

import numpy as np
import torch
import os
import faiss
import pandas as pd

# =============== 测评函数 ===============

from sklearn.metrics import average_precision_score

def compute_map(index, query_embs, query_ids, db_embs, db_ids, top_k=10):
    aps = []
    for i, q_emb in enumerate(query_embs):
        D, I = index.search(q_emb.reshape(1, -1).astype("float32"), top_k)
        retrieved_ids = [db_ids[j] for j in I[0]]
        gt = (np.array(retrieved_ids) == query_ids[i]).astype(int)
        if gt.sum() == 0:
            continue
        ap = average_precision_score(gt, D[0])
        aps.append(ap)
    mAP = np.mean(aps) if aps else 0
    return mAP

def zero_shot_topk_acc(index, query_embs, query_labels, db_embs, db_labels, topk=5):
    top1_correct = 0
    recallk = 0
    N = len(query_embs)
    for i, q_emb in enumerate(query_embs):
        D, I = index.search(q_emb.reshape(1, -1).astype("float32"), topk)
        topk_labels = [db_labels[j] for j in I[0]]
        gt = query_labels[i]
        if topk_labels[0] == gt:
            top1_correct += 1
        if gt in topk_labels:
            recallk += 1
    return top1_correct / N, recallk / N

def compute_bertscore(pred_captions, ref_captions, lang="en"):
    from bert_score import score
    P, R, F1 = score(pred_captions, ref_captions, lang=lang)
    return float(F1.mean())

def compute_bleu(pred_captions, ref_captions):
    from nltk.translate.bleu_score import corpus_bleu
    refs = [[r.split()] for r in ref_captions]
    hyps = [p.split() for p in pred_captions]
    bleu = corpus_bleu(refs, hyps)
    return bleu

# =============== 触觉投影工具 ===============

def project_all_tac_embeddings(tac_emb, projector, device):
    projector.eval()
    with torch.no_grad():
        tac_emb_tensor = torch.from_numpy(tac_emb).float().to(device)
        projected = projector(tac_emb_tensor).cpu().numpy()
    projected_norm = projected / np.linalg.norm(projected, axis=1, keepdims=True)
    return projected_norm

# =============== 主测评流程 ===============

if __name__ == "__main__":
    # -------- 配置部分 --------
    tactile_emb_dir = "embeddings/embeddings_tac"
    rgb_emb_dir = "embeddings/embeddings_rgb"
    projector_path = "tac_projector.pt"
    caption_csv = "data/ssvtp/new_train.csv"
    # =========================

    # ------- 加载主模型和embedding -------
    from initial2 import TouchQAModel  # 替换your_model_file为你的模型文件名

    os.environ[
        "OPENAI_API_KEY"] = "sk-proj"

    qa_model = TouchQAModel(
        tactile_emb_dir=tactile_emb_dir,
        rgb_emb_dir=rgb_emb_dir,
        projector_path=projector_path,
        caption_csv=caption_csv
    )
    device = qa_model.device

    print("已加载 TouchQAModel。")

    # ------- 投影tactile embedding -------
    tac_emb_projected = project_all_tac_embeddings(
        qa_model.tac_emb, qa_model.projector, device
    )
    tac_idx = qa_model.tac_idx
    tac_labels = tac_idx  # 若类别为index，否则替换为真实label

    # ------- RGB embedding -------
    rgb_emb_norm = qa_model.rgb_emb_norm
    rgb_idx = qa_model.rgb_idx
    rgb_labels = rgb_idx  # 若类别为index，否则替换为真实label

    # ------- 构建 faiss RGB 库 -------
    rgb_index = faiss.IndexFlatIP(rgb_emb_norm.shape[1])
    rgb_index.add(rgb_emb_norm.astype("float32"))

    # ------- 检索评测 -------
    print("正在计算 mAP ...")
    mAP = compute_map(
        rgb_index,
        tac_emb_projected, tac_idx,
        rgb_emb_norm, rgb_idx,
        top_k=10
    )
    print(f"MLP投影后 Tactile → RGB 检索 mAP: {mAP:.4f}")

    print("正在计算 Zero-shot 分类 Top-1/Recall@5 ...")
    top1, recall5 = zero_shot_topk_acc(
        rgb_index,
        tac_emb_projected, tac_labels,
        rgb_emb_norm, rgb_labels,
        topk=5
    )
    print(f"MLP投影后 Zero-shot 分类 Top-1: {top1:.4f} Recall@5: {recall5:.4f}")

    # ------- Caption测评 -------
    if os.path.exists(caption_csv):
        df = pd.read_csv(caption_csv, dtype={"index": str})
        ref_captions = df["caption"].tolist()

        # pred_captions = [你的模型批量生成的caption]
        # 这里用ref模拟，真实测评时请用模型批量生成的 pred_captions
        pred_captions = ref_captions

        try:
            bertscore = compute_bertscore(pred_captions, ref_captions)
            print(f"BERTScore F1: {bertscore:.4f}")
        except Exception as e:
            print(f"BERTScore 计算失败: {e}")

        try:
            bleu = compute_bleu(pred_captions, ref_captions)
            print(f"BLEU: {bleu:.4f}")
        except Exception as e:
            print(f"BLEU 计算失败: {e}")

    else:
        print("未找到caption csv，跳过caption评测。")

    print("全部评测完成！")

