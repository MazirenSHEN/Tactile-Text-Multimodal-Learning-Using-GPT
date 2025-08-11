import os

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "sk-proj"


from initial2 import TouchQAModel  # 替换为你的模型文件名

qa_model = TouchQAModel(
    tactile_emb_dir="embeddings/embeddings_tac",
    rgb_emb_dir="embeddings/embeddings_rgb",
    projector_path="tac_projector.pt",  # 没有MLP可写None
    caption_csv="data/ssvtp/new_train.csv",
    tactile_img_dir="data/ssvtp/images_tac",
    rgb_img_dir="data/ssvtp/images_rgb"
)

intention_questions = {
    "property": "Please describe the tactile properties of this sample.",
    "comparison": "Compared with rock, how does the tactile sensation of this sample differ?",
    "judgement": "Based on the tactile cues, what material do you think this sample is?",
}


df = pd.read_csv('data/test/test.csv')
if 'url' in df.columns:
    img_col = 'url'
elif 'image_path' in df.columns:
    img_col = 'image_path'
elif 'img' in df.columns:
    img_col = 'img'
else:
    raise Exception("无法识别图片路径列，请检查csv字段名。")

if 'label' in df.columns:
    label_col = 'label'
elif 'caption' in df.columns:
    label_col = 'caption'
else:
    raise Exception("无法识别label/caption列，请检查csv字段名。")

results = []
results2 = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_path = os.path.join('data/test', row[img_col].replace('\\', '/'))
    # image_path = row[img_col]
    label = row[label_col]

    # 预提embedding，自动识别图片类型
    try:
        emb = qa_model.extract_embedding(image_path)
        is_rgb = qa_model.is_rgb(emb)
        image_type = 'rgb' if is_rgb else 'tactile'
    except Exception as e:
        is_rgb = None
        image_type = f'[ERROR]: {str(e)}'

    for intent, question in intention_questions.items():
        # ---- 1. 检索topk reference label ----
        reference_labels = []
        if not is_rgb:  # tactile分支，才能检索reference label
            try:
                # 使用投影（如有）
                use_proj = qa_model.projector is not None
                emb_tac = qa_model.extract_embedding(image_path, use_projector=use_proj)
                intent_topk = qa_model.intent_topk.get(intent, 2)
                emb_norm = emb_tac / np.linalg.norm(emb_tac)
                D, I = qa_model.tac_index.search(emb_norm.reshape(1, -1).astype("float32"), intent_topk)
                for pos in I[0]:
                    idx_ref = qa_model.tac_idx[pos]
                    if qa_model.df is not None:
                        row_ref = qa_model.df[qa_model.df["index"] == idx_ref]
                        label_ref = row_ref.iloc[0]["caption"] if not row_ref.empty else "<unknown>"
                    else:
                        label_ref = "<unknown>"
                    reference_labels.append(label_ref)
            except Exception as e:
                reference_labels = [f"[ERROR]: {str(e)}"]
        else:
            reference_labels = []

        # ---- 2. 调用模型生成答案 ----
        try:
            answer = qa_model.answer(
                image_path,
                user_query=question
            )
            print(answer)
        except Exception as e:
            answer = f"[ERROR]: {str(e)}"
        results.append({
            "image_path": image_path,
            "label": label,
            "image_type": image_type,
            "intention": intent,
            "question": question,
            "reference_labels": "|".join(reference_labels),  # 用 | 分隔，Excel友好
            "model_answer": answer
        })
        results2.append({
            "label": label,
            "intention": intent,
            "question": question,
            "reference_labels": "|".join(reference_labels),  # 用 | 分隔，Excel友好
            "model_answer": answer
        })


out_df = pd.DataFrame(results)
out_df.to_csv("result2.csv", index=False, encoding="utf-8-sig")
out_df = pd.DataFrame(results2)
out_df.to_csv("result3.csv", index=False, encoding="utf-8-sig")

print("全部测评已完成，已保存 test1_multi_intent_eval_results_with_reference.csv")
