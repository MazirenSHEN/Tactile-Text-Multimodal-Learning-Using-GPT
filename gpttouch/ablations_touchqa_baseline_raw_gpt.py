#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TouchQA RAW Baseline (minimal, no prompt engineering, no project code)
---------------------------------------------------------------------
- Sends only [tactile image + question] directly to a base GPT model (default gpt-4o).
- Does NOT use any RAG / projector / template / qa_model.answer / custom system prompt from your project.
- Constructs the minimal Chat Completions message itself (user role only: one image + one text).
- If the CSV contains reference answers (label/caption/answer/ref), computes BLEU-4, Token-F1, and optionally BERTScore.
- Exports two result files matching your templates: result3.csv (full fields) and result4.csv (reduced fields), plus a summary.

Example:
  export OPENAI_API_KEY=sk-...
  python touchqa_raw_baseline_minimal_noprompt.py \
    --qa_csv data/ssvtp/test.csv \
    --tactile_img_root data/ssvtp \
    --out_dir ablation_outputs/raw_baseline \
    --model_name gpt-4o \
    --with_bertscore --bert_lang en  # use 'zh' for Chinese
"""

import os
import re
import json
import base64
import mimetypes
import argparse
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# optional: load .env
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# --- OpenAI client ---
from openai import OpenAI  # type: ignore

# --- metrics (fallback implementation) ---
try:
    from ablation_utils import bleu4 as _bleu4_u, token_f1 as _f1_u  # type: ignore
    _HAS_UTILS = True
except Exception:
    _HAS_UTILS = False

_ALNUM = re.compile(r"[A-Za-z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")

def _tok(s: str) -> List[str]:
    s = (s or "").strip().lower()
    toks = _ALNUM.findall(s)
    if toks:
        return toks
    cjk = _CJK.findall(s)
    ws = [t for t in re.split(r"\s+", s) if t]
    return cjk if len(cjk) >= len(ws) else ws

def bleu4(h: str, r: str) -> float:
    if _HAS_UTILS:
        return float(_bleu4_u(h, r))
    ht, rt = _tok(h), _tok(r)
    if not ht or not rt:
        return 0.0
    def cnt(tokens, n):
        d = {}
        for i in range(len(tokens)-n+1):
            ng = tuple(tokens[i:i+n]); d[ng] = d.get(ng, 0) + 1
        return d
    ps = []
    for n in (1,2,3,4):
        h, r = cnt(ht,n), cnt(rt,n)
        overlap = sum(min(c, r.get(ng,0)) for ng,c in h.items())
        total = max(1, sum(h.values()))
        p = overlap/total if total else 0.0
        if p == 0.0:
            p = 1.0/(2.0*total)
        ps.append(p)
    bp = 1.0 if len(ht) > len(rt) else np.exp(1 - len(rt)/max(1,len(ht)))
    return float(bp * np.prod([p**0.25 for p in ps]))

def token_f1(p: str, r: str) -> float:
    if _HAS_UTILS:
        return float(_f1_u(p, r))
    P, R = set(_tok(p)), set(_tok(r))
    if not P and not R:
        return 1.0
    if not P or not R:
        return 0.0
    tp = len(P & R)
    prec = tp/len(P)
    rec  = tp/len(R)
    return 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

def bootstrap_ci(vals: List[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 1234) -> Tuple[float,float]:
    if not vals:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=np.float64)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        boots.append(float(np.mean(arr[idx])))
    return float(np.quantile(boots, alpha/2)), float(np.quantile(boots, 1-alpha/2))

# --- BERTScore (optional) ---
_DEF_BERT_LANG = "en"
try:
    from bert_score import score as bert_score  # type: ignore
    _HAS_BERT = True
except Exception:
    _HAS_BERT = False

# --- image -> data URL ---

def image_to_data_url(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    if mt is None:
        mt = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mt};base64,{b64}"

# --- call GPT with only (image + question) ---

def ask_gpt_raw(image_path: str, question: str, model_name: str) -> str:
    client = OpenAI()
    data_url = image_to_data_url(image_path)
    # user message contains only: one image + one text; no system, no template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": question},
            ],
        }
    ]
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=128,
            temperature=0.7,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[ERROR] {e}"

# --- intentions (follow your template questions) ---
INTENTION_QUESTIONS = {
    "property":   "Please describe the tactile properties of this sample.",
    "comparison": "Compared with rock, how does this tactile sensation differ?",
    "judgement":  "Based on the tactile cues, what material do you think this sample is?",
}

# --- main ---

def main():
    ap = argparse.ArgumentParser("TouchQA RAW Baseline (no prompt engineering)")
    ap.add_argument("--qa_csv", type=str, default="data/ssvtp/test.csv")
    ap.add_argument("--tactile_img_root", type=str, default="data/ssvtp")
    ap.add_argument("--out_dir", type=str, default="ablation_outputs/raw_baseline")
    ap.add_argument("--model_name", type=str, default="gpt-4o")
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--with_bertscore", action="store_true")
    ap.add_argument("--bert_lang", type=str, default=_DEF_BERT_LANG)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.qa_csv)
    if args.max_rows and args.max_rows > 0 and len(df) > args.max_rows:
        df = df.head(args.max_rows).copy()
        print(f"[info] quick run: top-{args.max_rows} rows")

    # image column (follow your field priority)
    if 'tactile' in df.columns:
        img_col = 'tactile'
    elif 'image_path' in df.columns:
        img_col = 'image_path'
    elif 'img' in df.columns:
        img_col = 'img'
    else:
        raise Exception("Cannot detect image path column. Please check CSV column names (tactile/image_path/img)")

    # label column (optional)
    if 'label' in df.columns:
        label_col = 'label'
    elif 'caption' in df.columns:
        label_col = 'caption'
    elif 'answer' in df.columns:
        label_col = 'answer'
    elif 'ref' in df.columns:
        label_col = 'ref'
    else:
        label_col = None
        print("[warn] No reference answer column found (label/caption/answer/ref); will only record answers and won't compute scores.")

    results = []  # corresponds to your original result3.csv (full fields)
    results2 = [] # corresponds to your original result4.csv (reduced)
    hyps, refs = [], []  # for overall BERTScore

    calls = 0
    n_err = 0
    n_empty = 0

    t0 = time.perf_counter()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        rel = str(row[img_col])
        image_path = os.path.join(args.tactile_img_root, rel.replace('\\', '/'))
        if not os.path.exists(image_path):
            continue
        label = str(row[label_col]) if (label_col and isinstance(row[label_col], str)) else ""

        for intent, question in INTENTION_QUESTIONS.items():
            calls += 1
            answer = ask_gpt_raw(image_path, question, args.model_name)
            if not isinstance(answer, str) or len(answer.strip()) == 0 or answer.strip().lower().startswith('[error]'):
                n_empty += 1
            if answer.startswith('[ERROR]'):
                n_err += 1

            b4 = bleu4(answer, label) if label else 0.0
            f1 = token_f1(answer, label) if label else 0.0

            results.append({
                "image_path": image_path,
                "label": label,
                "image_type": "tactile",   # baseline: fixed
                "intention": intent,
                "question": question,
                "reference_labels": "",     # baseline does not provide references
                "model_answer": answer,
                "BLEU4": b4,
                "TokenF1": f1,
            })
            results2.append({
                "label": label,
                "intention": intent,
                "question": question,
                "reference_labels": "",
                "model_answer": answer,
                "BLEU4": b4,
                "TokenF1": f1,
            })

            if label:
                refs.append(label)
                hyps.append(answer)

    t1 = time.perf_counter()
    elapsed = t1 - t0

    # save per-sample rows
    out3 = os.path.join(args.out_dir, "resultbaseline1.csv")
    out4 = os.path.join(args.out_dir, "resultbaseline2.csv")
    pd.DataFrame(results).to_csv(out3, index=False, encoding="utf-8-sig")
    pd.DataFrame(results2).to_csv(out4, index=False, encoding="utf-8-sig")

    # summary + BERTScore (optional)
    bleu_vals = [float(x) for x in pd.DataFrame(results)["BLEU4"].dropna().tolist()] if label_col else []
    f1_vals   = [float(x) for x in pd.DataFrame(results)["TokenF1"].dropna().tolist()] if label_col else []
    b_lo, b_hi = bootstrap_ci(bleu_vals) if bleu_vals else (0.0, 0.0)
    f_lo, f_hi = bootstrap_ci(f1_vals)   if f1_vals else (0.0, 0.0)

    stats = {
        "N_rows": int(len(results)),
        "LLM_calls": int(calls),
        "TotalSeconds": float(elapsed),
        "MeanSecPerCall": float(elapsed / max(1, calls)),
        "QPS": float(calls / elapsed) if elapsed > 0 else 0.0,
        "ErrorRate": float(n_err / max(1, calls)),
        "EmptyAnswerRate": float(n_empty / max(1, calls)),
        "BLEU4_mean": float(np.mean(bleu_vals)) if bleu_vals else 0.0,
        "BLEU4_CI_low": b_lo,
        "BLEU4_CI_high": b_hi,
        "TokenF1_mean": float(np.mean(f1_vals)) if f1_vals else 0.0,
        "TokenF1_CI_low": f_lo,
        "TokenF1_CI_high": f_hi,
    }

    if args.with_bertscore and hyps and refs:
        if _HAS_BERT:
            try:
                P, R, F = bert_score(hyps, refs, lang=args.bert_lang, rescale_with_baseline=True)
                stats.update({
                    "BERT_P": float(P.mean().item()),
                    "BERT_R": float(R.mean().item()),
                    "BERT_F1": float(F.mean().item()),
                })
            except Exception as e:
                stats["BERTScore_note"] = f"failed: {e}"
        else:
            stats["BERTScore_note"] = "bert-score not installed; pip install bert-score"

    out_json = os.path.join(args.out_dir, "raw_baseline_summary.json")
    out_csv  = os.path.join(args.out_dir, "raw_baseline_summary.csv")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats,f, ensure_ascii=False, indent=2)
    pd.DataFrame([stats]).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("== RAW Baseline Summary ==")
    print(f"Saved rows -> {out3} & {out4}")
    print(f"Saved summary -> {out_json} & {out_csv}")

if __name__ == "__main__":
    main()
