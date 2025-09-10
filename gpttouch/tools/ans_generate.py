import os
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from TactileQASystem_integrated import TouchQAModel  # replace with your model filename

qa_model = TouchQAModel(
    tactile_emb_dir="embeddings/embeddings_tac",   # kept for compatibility, model no longer uses tac indices
    rgb_emb_dir="embeddings/embeddings_rgb",
    projector_path="tac_projector_vit5p_best.pt",  # set to None if no projector
    caption_csv="data/ssvtp/new_train.csv",
    tactile_img_dir="data/ssvtp/images_tac",
    rgb_img_dir="data/ssvtp/images_rgb"
)

intention_questions = {
    "property": "Please describe the tactile properties of this sample.",
    "comparison": "Compared with rock, how does the tactile sensation of this sample differ?",
    "judgement": "Based on the tactile cues, what material do you think this sample is?",
}

df = pd.read_csv('data/ssvtp/test.csv')
if 'tactile' in df.columns:
    img_col = 'tactile'
elif 'image_path' in df.columns:
    img_col = 'image_path'
elif 'img' in df.columns:
    img_col = 'img'
else:
    raise Exception("Could not identify image path column; please check CSV field names.")

if 'label' in df.columns:
    label_col = 'label'
elif 'caption' in df.columns:
    label_col = 'caption'
else:
    raise Exception("Could not identify label/caption column; please check CSV field names.")

results = []
results2 = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_path = os.path.join('data/ssvtp', str(row[img_col]).replace('\\', '/'))
    label = row[label_col]

    # Pre-extract embedding and determine image type (optional)
    try:
        emb_raw = qa_model.extract_raw_embedding(image_path)
        is_rgb = qa_model.is_rgb(emb_raw)
        image_type = 'rgb' if is_rgb else 'tactile'
    except Exception as e:
        emb_raw = None
        is_rgb = None
        image_type = f'[ERROR]: {str(e)}'

    for intent, question in intention_questions.items():
        # ---- 1) Use tac->rgb retrieval to get reference labels (take captions from RGB neighbors' IDs) ----
        reference_labels = []
        try:
            if emb_raw is None:
                # If embedding extraction failed above, try again here
                emb_raw = qa_model.extract_raw_embedding(image_path)

            # If a projector exists, map the query vector to RGB space; otherwise use the raw vector
            if qa_model.projector is not None:
                q_vec = qa_model.apply_projector_to_vector(emb_raw)
            else:
                q_vec = emb_raw

            intent_topk = qa_model.intent_topk.get(intent, 2)
            rgb_neighbors = qa_model._search_ids(qa_model.rgb_index, q_vec, intent_topk)

            # Map rgb neighbor IDs to captions via idx2caption
            for rid, _score in rgb_neighbors:
                # use string ID lookup; fallback to <unknown> if missing
                cap = qa_model.idx2caption.get(str(rid), "<unknown>")
                reference_labels.append(cap)

        except Exception as e:
            reference_labels = [f"[ERROR]: {str(e)}"]

        # ---- 2) Ask the model to answer (internally it may also use the same tac->rgb references) ----
        try:
            answer = qa_model.answer(
                image_path,
                user_query=question
            )
            # print(answer)  # uncomment to debug / inspect
        except Exception as e:
            answer = f"[ERROR]: {str(e)}"

        results.append({
            "image_path": image_path,
            "label": label,
            "image_type": image_type,
            "intention": intent,
            "question": question,
            "reference_labels": "|".join(reference_labels),  # separated by |
            "model_answer": answer
        })
        results2.append({
            "label": label,
            "intention": intent,
            "question": question,
            "reference_labels": "|".join(reference_labels),
            "model_answer": answer
        })

# Save results
pd.DataFrame(results).to_csv("result3.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(results2).to_csv("result4.csv", index=False, encoding="utf-8-sig")
print("All evaluations completed; saved result3.csv and result4.csv")
