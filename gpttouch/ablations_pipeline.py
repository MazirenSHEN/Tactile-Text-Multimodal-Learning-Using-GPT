#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablations_pipeline.py (OOM-safe, Stage2 skips projector)
"""
import os
# 1) Reduce CUDA memory fragmentation (must set before any torch-related import)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc  # for explicit garbage collection

from TactileQASystem_integrated import TouchQAModel
from ablation_utils import bleu4, token_f1, try_bertscore

# Default LLM (used if TouchQAModel doesn't provide ft_model_name)
DEFAULT_FT_MODEL = os.getenv("FT_MODEL_NAME", "gpt-4o-mini")

INTENTION_QUESTIONS = {
    "property":   "Please describe the tactile properties of this sample.",
    "comparison": "Compared with rock, how does the tactile sensation of this sample differ?",
    "judgement":  "Based on the tactile cues, what material do you think this sample is?",
}

def load_gold_answers_jsonl(path):
    if not path or not os.path.exists(path):
        return {}
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            ans = j.get("gold") or j.get("answer") or j.get("label") or j.get("caption")
            if not isinstance(ans, str):
                continue
            tid = None
            for k in ["image_path", "tactile", "img", "path"]:
                if j.get(k):
                    tid = _extract_tid_from_path(str(j[k])); break
            if tid is None and j.get("tid") is not None:
                try: tid = int(j["tid"])
                except: tid = None
            if tid is not None:
                m[tid] = ans
    return m

def _extract_tid_from_path(p: str):
    if not isinstance(p, str):
        return None
    p = p.replace("\\", "/")
    import re, os as _os
    m = re.search(r"image_(\d+)_(?:tac|rgb)\.(?:jpg|jpeg|png|bmp|webp)$", p, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", _os.path.basename(p))
    return int(m2.group(1)) if m2 else None

def ablation_answer(
    qa_model: TouchQAModel,
    image_path: str,
    user_query: str,
    use_rgb_refs: bool,
    rgb_source: str,  # 'projector' | 'paired' | 'none'
    use_intent_templates: bool,
):
    import base64, os
    emb_raw = qa_model.extract_raw_embedding(image_path)
    intent = qa_model.classify_intent(user_query)
    topk = getattr(qa_model, "intent_topk", {}).get(intent or "other", 2)

    # tactile neighbors -> captions
    tac_neighbors = qa_model._search_ids(qa_model.tac_index, emb_raw, topk)
    ref_labels = []
    if getattr(qa_model, "df", None) is not None and getattr(qa_model, "idx2caption", None) is not None:
        for (tid, _) in tac_neighbors:
            key = str(tid)
            label = qa_model.idx2caption.get(key, "<unknown>")
            ref_labels.append(label)

    # RGB refs (only use projector when needed)
    ref_rgb_paths = []
    if use_rgb_refs and rgb_source != "none":
        if rgb_source == "projector" and getattr(qa_model, "projector", None) is not None:
            emb_proj = qa_model.apply_projector_to_vector(emb_raw)
            rgb_neighbors = qa_model._search_ids(qa_model.rgb_index, emb_proj, topk)
            for (rid, _) in rgb_neighbors:
                rp = os.path.join(qa_model.rgb_img_dir, f"image_{rid}_rgb.jpg")
                if os.path.exists(rp):
                    ref_rgb_paths.append(rp)
        elif rgb_source == "paired":
            for (tid, _) in tac_neighbors:
                rp = os.path.join(qa_model.rgb_img_dir, f"image_{tid}_rgb.jpg")
                if os.path.exists(rp):
                    ref_rgb_paths.append(rp)

    captions_text = "; ".join([f"{i+1}. {c}" for i, c in enumerate(ref_labels)])

    # prompts
    if use_intent_templates:
        intent_instruction_map = {
            "property": ("Focus on touch-based attributes. Avoid visual-only cues."),
            "comparison": ("Compare the target sample with the reference tactile labels. Emphasize differences/similarities in tactile qualities."),
            "judgement": ("Infer the possible material from tactile cues. Express uncertainty if needed."),
            "other": ("Use only tactile reasoning; rely on reference tactile labels if helpful."),
        }
        task_instruction = intent_instruction_map.get(intent or "other", intent_instruction_map["other"])
        prompt = (
            f"You are given a tactile sensor image.\n"
            f"Reference tactile labels:\n{captions_text}\n"
            f"{task_instruction}\n"
            f"User Question:{user_query}\n"
        )
        system_role = "You are a tactile perception expert. Keep answers <=30 words."
    else:
        prompt = user_query
        system_role = "You are a tactile assistant. Provide a brief, precise answer (<=2 sentences)."

    # assemble messages
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    user_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}},
    ]
    for rp in ref_rgb_paths:
        with open(rp, "rb") as f:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}", "detail": "low"}
            })

    messages = [{"role": "system", "content": system_role},
                {"role": "user", "content": user_content}]

    _model_name = getattr(qa_model, "ft_model_name", None) or DEFAULT_FT_MODEL
    resp = qa_model.client.chat.completions.create(
        model=_model_name,
        messages=messages,
        max_tokens=128,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip(), len(ref_rgb_paths), bool(use_intent_templates)

def make_model(tactile_emb_dir, rgb_emb_dir, projector_path, caption_csv, tactile_img_root, ft_model_name):
    kwargs = dict(
        tactile_emb_dir=tactile_emb_dir,
        rgb_emb_dir=rgb_emb_dir,
        projector_path=projector_path,  # can be None
        caption_csv=caption_csv,
        tactile_img_dir=os.path.join(tactile_img_root, "images_tac"),
        rgb_img_dir=os.path.join(tactile_img_root, "images_rgb"),
    )
    try:
        m = TouchQAModel(**kwargs, ft_model_name=ft_model_name)
    except TypeError:
        m = TouchQAModel(**kwargs)
        setattr(m, "ft_model_name", ft_model_name)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_csv", type=str, default="data/ssvtp/test.csv")
    ap.add_argument("--tactile_emb_dir", type=str, default="embeddings/embeddings_tac")
    ap.add_argument("--rgb_emb_dir", type=str, default="embeddings/embeddings_rgb")
    ap.add_argument("--caption_csv", type=str, default="data/ssvtp/new_train.csv")
    ap.add_argument("--tactile_img_root", type=str, default="data/ssvtp")
    ap.add_argument("--out_dir", type=str, default="ablation_outputs/pipeline")

    # default projector (fallback when a specific setting doesn't set one)
    ap.add_argument("--projector_path", type=str, default="tac_projector_vit5p_best.pt")

    # per-setting projectors (optional)
    ap.add_argument("--projector_stage1", type=str, default=None)
    ap.add_argument("--projector_stage2", type=str, default=None)
    ap.add_argument("--projector_full",  type=str, default=None)

    # per-setting LLM names (optional)
    ap.add_argument("--ft_model_stage1", type=str, default=None)
    ap.add_argument("--ft_model_stage2", type=str, default="ft:gpt-4o-2024-08-06:personal:tactilemode:C3Y5BzQp")
    ap.add_argument("--ft_model_full",  type=str, default="ft:gpt-4o-2024-08-06:personal:tactilemode:C3Y5BzQp")

    ap.add_argument("--gold_jsonl", type=str, default="test_answers.jsonl")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.qa_csv)
    if 'tactile' in df.columns:
        img_col = 'tactile'
    elif 'image_path' in df.columns:
        img_col = 'image_path'
    elif 'img' in df.columns:
        img_col = 'img'
    else:
        raise RuntimeError("Cannot find tactile image column.")

    if 'label' in df.columns:
        label_col = 'label'
    elif 'caption' in df.columns:
        label_col = 'caption'
    else:
        label_col = None

    gold_map = load_gold_answers_jsonl(args.gold_jsonl) if args.gold_jsonl else {}

    # 2) Key: Stage2 does not need a projector, force None to avoid GPU warm-up
    settings = {
        "Stage1_only_multimodal": dict(
            use_rgb_refs=True, rgb_source="projector", use_intent_templates=True,
            projector=args.projector_stage1 or args.projector_path,
            ft_model=args.ft_model_stage1 or DEFAULT_FT_MODEL,
        ),
        "Stage2_only_tactile": dict(
            use_rgb_refs=False, rgb_source="none", use_intent_templates=False,
            projector=None,  # <<< do not load projector
            ft_model=args.ft_model_stage2 or DEFAULT_FT_MODEL,
        ),
        "Full_Stage1_to_Stage2": dict(
            use_rgb_refs=True, rgb_source="projector", use_intent_templates=True,
            projector=args.projector_full or args.projector_path,
            ft_model=args.ft_model_full or DEFAULT_FT_MODEL,
        ),
    }

    all_summaries = []

    for name, cfg in settings.items():
        print(f"\n==== Running setting: {name} ====")
        print(f"[CFG] projector={cfg['projector']}")
        print(f"[CFG] ft_model={cfg['ft_model']}")

        # initialize model
        qa_model = make_model(
            tactile_emb_dir=args.tactile_emb_dir,
            rgb_emb_dir=args.rgb_emb_dir,
            projector_path=cfg["projector"],
            caption_csv=args.caption_csv,
            tactile_img_root=args.tactile_img_root,
            ft_model_name=cfg["ft_model"],
        )
        if not hasattr(qa_model, "ft_model_name"):
            setattr(qa_model, "ft_model_name", cfg["ft_model"])

        per_rows = []
        hyps, refs = [], []
        rgb_attached_total = 0
        used_template_total = 0

        for _, r in tqdm(df.iterrows(), total=len(df)):
            tac_rel = r[img_col]
            image_path = os.path.join(args.tactile_img_root, str(tac_rel).replace("\\", "/"))
            if not os.path.exists(image_path):
                continue

            for intent, q in INTENTION_QUESTIONS.items():
                try:
                    ans, rgb_cnt, used_tpl = ablation_answer(
                        qa_model, image_path, q,
                        use_rgb_refs=cfg["use_rgb_refs"],
                        rgb_source=cfg["rgb_source"],
                        use_intent_templates=cfg["use_intent_templates"],
                    )
                except Exception as e:
                    ans, rgb_cnt, used_tpl = f"[ERROR]: {e}", 0, False

                tid = _extract_tid_from_path(image_path)
                if label_col and isinstance(r.get(label_col), str):
                    ref = r.get(label_col)
                else:
                    ref = gold_map.get(tid, "")

                b4 = bleu4(ans, ref) if ref else 0.0
                f1 = token_f1(ans, ref) if ref else 0.0

                per_rows.append({
                    "image_path": image_path,
                    "intent": intent,
                    "question": q,
                    "reference_label": ref,
                    "model_answer": ans,
                    "rgb_refs_attached": rgb_cnt,
                    "used_intent_template": used_tpl,
                    "BLEU4": b4,
                    "TokenF1": f1,
                })
                rgb_attached_total += rgb_cnt
                used_template_total += int(used_tpl)

                if ref:
                    hyps.append(ans); refs.append(ref)

        # BERTScore
        bert_res, bert_err = try_bertscore(hyps, refs) if len(refs) > 0 else (None, None)

        # save details
        detail_csv = os.path.join(args.out_dir, f"{name}_detail.csv")
        pd.DataFrame(per_rows).to_csv(detail_csv, index=False, encoding="utf-8-sig")

        df_detail = pd.DataFrame(per_rows)
        bleu_mean = float(df_detail["BLEU4"].mean()) if not df_detail.empty else 0.0
        f1_mean   = float(df_detail["TokenF1"].mean()) if not df_detail.empty else 0.0

        # summary
        summ = {
            "setting": name,
            "N_samples": int(len(per_rows)),
            "BLEU4_mean": bleu_mean,
            "TokenF1_mean": f1_mean,
            "rgb_refs_attached_total": int(rgb_attached_total),
            "used_template_total": int(used_template_total),
            "projector_path": cfg["projector"],
            "ft_model_name": cfg["ft_model"],
        }
        if bert_res:
            summ.update({f"BERTScore_{k}": v for k, v in bert_res.items()})
        if bert_err:
            summ["BERTScore_note"] = bert_err

        all_summaries.append(summ)

        print(f"Setting {name} -> BLEU4_mean={bleu_mean:.4f}, TokenF1_mean={f1_mean:.4f}")
        if bert_res:
            print(f" BERTScore F1={bert_res['F1']:.4f}")
        print(f" Attached RGB refs total: {rgb_attached_total}, Used templates total: {used_template_total}")

        # 3) Key: after each setting, proactively free memory to prevent OOM on the next run
        try:
            del qa_model
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    summary_csv = os.path.join(args.out_dir, "summary.csv")
    pd.DataFrame(all_summaries).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\nAll done. Summary saved -> {summary_csv}")

if __name__ == "__main__":
    print("[DEBUG] entered ablations_pipeline.main()")
    main()
