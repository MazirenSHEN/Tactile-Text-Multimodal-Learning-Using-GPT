#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TouchQA Ablation Runner (9-round design)
----------------------------------------
- Fixed 8–9 small-grid rounds (optional oracle included)
- Evaluation premise: same batch of samples; you can sample N=50 first; ensure label/caption columns exist
- Default hyperparams: intent=gold, rgbimg=0, intentTpl=1 (unless the factor under study changes them)
- Metrics: BLEU-4, Token-F1 (per-sample), optional BERTScore
- Statistics: 1000 bootstrap iterations for 95% CI on BLEU/F1 (ablation_utils.bootstrap_ci)
- Outputs: per-round per-row CSV + summary.csv (includes ΔBLEU, ΔF1, time/cost, failure rate)

Round definitions
-----------------
B0 Baseline: tac2tac, rgb=0, tpl=1, intent=gold
A1 Projector: tac2rgb_projector, 0, 1, gold
A2 Upper bound (optional): tac2rgb_paired, 0, 1, gold  (requires ID pairing; enabled with --include_oracle)
B1 Template effect: tac2tac, 0, 0, gold
C1 Feed RGB: tac2tac, 1, 1, gold
D1 Intent prediction instead of human: tac2tac, 0, 1, predicted
E1 Interaction: tac2rgb_projector, 0, 0, gold
E2 Interaction: tac2rgb_projector, 1, 1, gold

Example usage (baseline + other 7 rounds => 8 rounds; add --include_oracle for 9 rounds)
  python ablations_touchqa.py \
    --qa_csv data/ssvtp/test.csv \
    --tactile_img_root data/ssvtp \
    --out_dir ablation_outputs/touchqa9 \
    --max_rows 50 \
    --skip_bertscore
"""

import os
import re
import argparse
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Reduce noise
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

# --- Import compatibility (use TactileQASystem in your environment; otherwise raise) ---
try:
    from TactileQASystem_integrated import TouchQAModel
except Exception as e:  # pragma: no cover
    TouchQAModel = None
    _TQS_IMPORT_ERR = e
else:
    _TQS_IMPORT_ERR = None

# ---- Use shared utilities (includes bootstrap_ci / optional BERTScore) ----
try:
    from ablation_utils import bootstrap_ci, try_bertscore, bleu4 as _bleu4_u, token_f1 as _f1_u  # :contentReference[oaicite:2]{index=2}
    _USE_UTILS_METRICS = True
except Exception:
    _USE_UTILS_METRICS = False

# -----------------------------
# Text and intents
# -----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def _tok(s: str) -> List[str]:
    s = (s or "").lower()
    return _WORD_RE.findall(s)

INTENTION_QUESTIONS = {
    "property": "Based on tactile cues only, describe the key touch properties (e.g., roughness, hardness, compliance).",
    "comparison": "Compare this sample's tactile sensation with the reference tactile labels. Focus on touch attributes only.",
    "judgement": "From tactile characteristics alone, what is the likely material or physical state? State uncertainty if needed.",
}

# Built-in fallback metrics (used if ablation_utils cannot be imported)
def bleu4(hyp: str, ref: str) -> float:
    if _USE_UTILS_METRICS:
        return _bleu4_u(hyp, ref)
    hyp_t = _tok(hyp); ref_t = _tok(ref)
    if not hyp_t or not ref_t:
        return 0.0
    precisions = []
    for n in range(1, 5):
        # clipped precision
        def _count_ngrams(tokens, n):
            d = {}
            for i in range(len(tokens) - n + 1):
                ng = tuple(tokens[i:i+n]); d[ng] = d.get(ng, 0) + 1
            return d
        h = _count_ngrams(hyp_t, n); r = _count_ngrams(ref_t, n)
        overlap = sum(min(c, r.get(ng, 0)) for ng, c in h.items())
        total = max(1, sum(h.values()))
        p_n = overlap / total if total else 0.0
        if p_n == 0.0:  # simple smoothing
            p_n = 1.0 / (2.0 * total)
        precisions.append(p_n)
    # BP
    rl, hl = len(ref_t), len(hyp_t)
    bp = 1.0 if hl > rl else np.exp(1 - rl / max(1, hl))
    return float(bp * np.prod([p ** 0.25 for p in precisions]))

def token_f1(pred: str, ref: str) -> float:
    if _USE_UTILS_METRICS:
        return _f1_u(pred, ref)
    p = set(_tok(pred)); r = set(_tok(ref))
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    tp = len(p & r); prec = tp / len(p); rec = tp / len(r)
    return 0.0 if (prec + rec == 0) else 2 * prec * rec / (prec + rec)

# -----------------------------
# Column names & TID parsing
# -----------------------------
IMG_COL_CANDIDATES = ["image_path", "image", "img", "tactile_image", "img_path"]
LABEL_COL_CANDIDATES = ["label", "caption", "answer", "ref"]
ID_PAT = re.compile(r"image_(\d+)\.(?:jpg|jpeg|png|bmp|webp)$", re.IGNORECASE)

def parse_tid_from_path(path: str) -> Optional[int]:
    if not path:
        return None
    fname = os.path.basename(path)
    m = ID_PAT.search(fname)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    nums = re.findall(r"(\d+)", fname)
    return int(nums[-1]) if nums else None

# -----------------------------
# Safely pass params to qa_model.answer (compatible with different signatures)
# -----------------------------
import inspect

def call_answer(qa_model, image_path: str, user_query: str, **kwargs):
    fn = getattr(qa_model, "answer", None)
    if fn is None:
        raise RuntimeError("qa_model has no method 'answer'")
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(image_path, user_query, **filtered)

# -----------------------------
# Unified wrapper (keeps parity with original)
# -----------------------------
def ablation_answer_touchqa_with_intent(
    qa_model,
    image_path: str,
    user_query: str,
    *,
    retrieval_source: str,
    with_rgb_images: bool,
    use_intent_templates_flag: bool,
    intent_mode: str,
    gold_intent_map: Optional[Dict[int, str]] = None,
    tid: Optional[int] = None,
):
    """
    Returns (answer, ref_labels, predicted_intent, gold_intent, used_intent)
    """
    gold_intent = None
    if intent_mode == "gold" and gold_intent_map is not None and tid is not None:
        gold_intent = gold_intent_map.get(int(tid))
    params = dict(
        retrieval_source=retrieval_source,
        with_rgb_images=with_rgb_images,
        use_professional_prompt=use_intent_templates_flag,
        intent_mode=intent_mode,
        gold_intent_map=gold_intent_map,
        sample_tid=tid,
    )
    out = call_answer(qa_model, image_path, user_query, **params)

    ans, ref_labels, pred_intent, used_intent = None, [], None, None
    if isinstance(out, tuple):
        if len(out) == 5:
            ans, ref_labels, pred_intent, gold_from_model, used_intent = out
            if gold_from_model is not None:
                gold_intent = gold_from_model
        elif len(out) == 4:
            ans, ref_labels, pred_intent, used_intent = out
        elif len(out) == 3:
            ans, ref_labels, pred_intent = out
        elif len(out) >= 1:
            ans = out[0]
    elif isinstance(out, str):
        ans = out
    else:
        ans = str(out)
    return ans or "", (ref_labels or []), pred_intent, gold_intent, (used_intent or intent_mode)

# -----------------------------
# Round configuration (8; add A2 for 9 with --include_oracle)
# -----------------------------
_SRC_ABBR = {"tac2tac":"t2t", "tac2rgb_projector":"t2rgbP", "tac2rgb_paired":"t2rgbGT"}

def build_9rounds(include_oracle: bool) -> List[Dict]:
    rounds = [
        {"key":"B0", "desc":"baseline",      "src":"tac2tac",            "rgb":0, "tpl":1, "imode":"gold"},
        {"key":"A1", "desc":"projector",     "src":"tac2rgb_projector",  "rgb":0, "tpl":1, "imode":"gold"},
        {"key":"B1", "desc":"no_template",   "src":"tac2tac",            "rgb":0, "tpl":0, "imode":"gold"},
        {"key":"C1", "desc":"with_rgb",      "src":"tac2tac",            "rgb":1, "tpl":1, "imode":"gold"},
        {"key":"D1", "desc":"intent_pred",   "src":"tac2tac",            "rgb":0, "tpl":1, "imode":"predicted"},
        {"key":"E1", "desc":"proj_x_noTpl",  "src":"tac2rgb_projector",  "rgb":0, "tpl":0, "imode":"gold"},
        {"key":"E2", "desc":"proj_x_rgb",    "src":"tac2rgb_projector",  "rgb":1, "tpl":1, "imode":"gold"},
    ]
    if include_oracle:
        rounds.insert(2, {"key":"A2", "desc":"oracle_paired", "src":"tac2rgb_paired", "rgb":0, "tpl":1, "imode":"gold"})
    return rounds

def setting_name(cfg: Dict) -> str:
    return f"{cfg['key']}__{_SRC_ABBR.get(cfg['src'], cfg['src'])}__rgb{cfg['rgb']}__tpl{cfg['tpl']}__imode_{cfg['imode']}"

# -----------------------------
# Main flow
# -----------------------------
def main():
    ap = argparse.ArgumentParser("TouchQA Ablations (9 rounds)")
    ap.add_argument("--qa_csv", type=str, default="data/ssvtp/test.csv", help="CSV with test samples")
    ap.add_argument("--tactile_img_root", type=str, default="data/ssvtp", help="Root directory for tactile images")
    ap.add_argument("--out_dir", type=str, default="ablation_outputs/touchqa", help="Output directory")

    # Model / data knobs (passed through)
    ap.add_argument("--tactile_emb_dir", type=str, default="embeddings/embeddings_tac")
    ap.add_argument("--rgb_emb_dir", type=str, default="embeddings/embeddings_rgb")
    ap.add_argument("--caption_csv", type=str, default="data/ssvtp/new_train.csv")
    ap.add_argument("--projector_path", type=str, default="tac_projector_vit5_best.pt")

    # Gold intents (optional JSONL: {"tid": int, "intent": str})
    ap.add_argument("--gold_intent_jsonl", type=str, default="")

    # Evaluation control
    ap.add_argument("--max_rows", type=int, default=0, help=">0: use top-N rows (quick run)")
    ap.add_argument("--include_oracle", action="store_true", help="Include A2 (tac2rgb_paired) for 9 rounds")
    ap.add_argument("--skip_bertscore", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.qa_csv)
    if args.max_rows and args.max_rows > 0 and len(df) > args.max_rows:
        df = df.head(args.max_rows).copy()
        print(f"[info] Using top-{args.max_rows} rows for quick run.")

    # Column detection
    img_col = next((c for c in IMG_COL_CANDIDATES if c in df.columns), None)
    if img_col is None:
        img_col = df.columns[0]
        print(f"[warn] image column not found; falling back to '{img_col}'")

    label_col = next((c for c in LABEL_COL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        print("[warn] label/caption column not found; BLEU/F1 may be near 0.")

    # Load gold intent (optional)
    gold_intent_map: Dict[int, str] = {}
    if args.gold_intent_jsonl and os.path.exists(args.gold_intent_jsonl):
        with open(args.gold_intent_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    tid = int(obj.get("tid"))
                    intent = str(obj.get("intent", "")).strip()
                    if intent:
                        gold_intent_map[tid] = intent
                except Exception:
                    continue
        print(f"[info] Loaded gold intents: {len(gold_intent_map)} entries")

    # Build QA model
    if TouchQAModel is None:
        raise RuntimeError(f"Failed to import TactileQASystem: {_TQS_IMPORT_ERR}")

    qa_model = TouchQAModel(
        tactile_emb_dir=args.tactile_emb_dir,
        rgb_emb_dir=args.rgb_emb_dir,
        projector_path=args.projector_path,
        caption_csv=args.caption_csv,
    )
    # Optionally inject a tactile_img_root attribute if present (harmless if unsupported)
    if hasattr(qa_model, "tactile_img_root"):
        qa_model.tactile_img_root = args.tactile_img_root

    # Rounds (8 or 9)
    rounds = build_9rounds(args.include_oracle)
    print(f"[info] Planned rounds: {len(rounds)}")

    # Summary accumulator
    all_summaries = []
    baseline_stats = None  # used for Δ metrics

    for cfg in rounds:
        name = setting_name(cfg)
        print(f"\n==== Setting: {name} ====")

        per_rows = []
        hyps, refs = [], []
        intents_ok = []
        n_calls = 0
        n_errors = 0
        n_empty = 0

        t0 = time.perf_counter()

        for _, r in tqdm(df.iterrows(), total=len(df)):
            tac_rel = r[img_col]
            img_path = os.path.join(args.tactile_img_root, str(tac_rel).replace("\\", "/"))
            if not os.path.exists(img_path):
                continue

            # TID
            tid = None
            if "tid" in r and not pd.isna(r["tid"]):
                try:
                    tid = int(r["tid"])
                except Exception:
                    tid = parse_tid_from_path(str(tac_rel))
            else:
                tid = parse_tid_from_path(str(tac_rel))

            for intent_name, q in INTENTION_QUESTIONS.items():
                n_calls += 1
                try:
                    ans, ref_labels, pred_intent, gold_intent, used_intent = ablation_answer_touchqa_with_intent(
                        qa_model,
                        img_path,
                        q,
                        retrieval_source=cfg["src"],
                        with_rgb_images=bool(cfg["rgb"]),
                        use_intent_templates_flag=bool(cfg["tpl"]),
                        intent_mode=cfg["imode"],
                        gold_intent_map=gold_intent_map if gold_intent_map else None,
                        tid=tid,
                    )
                    intent_correct = (gold_intent is not None) and (pred_intent == gold_intent)
                except Exception as e:
                    ans, ref_labels, pred_intent, gold_intent, used_intent = f"[ERROR]: {e}", [], None, None, None
                    intent_correct = False
                    n_errors += 1

                # Track failures / empty answers
                if not isinstance(ans, str) or len(ans.strip()) == 0 or ans.strip().lower().startswith("[error]"):
                    n_empty += 1

                ref = r[label_col] if label_col and isinstance(r[label_col], str) else ""
                b4 = bleu4(ans, ref) if ref else 0.0
                f1 = token_f1(ans, ref) if ref else 0.0

                per_rows.append({
                    "setting": name,
                    "key": cfg["key"],
                    "desc": cfg["desc"],
                    "image_rel": str(tac_rel),
                    "image_path": img_path,
                    "tid": tid,
                    "intent_qtype": intent_name,
                    "intent_mode": cfg["imode"],
                    "used_intent": used_intent,
                    "pred_intent": pred_intent,
                    "gold_intent": gold_intent,
                    "intent_correct": int(bool(intent_correct)),
                    "answer": ans,
                    "reference_labels": "|".join(ref_labels) if ref_labels else "",
                    "label": ref,
                    "BLEU4": b4,
                    "TokenF1": f1,
                })

                if ref:
                    hyps.append(ans)
                    refs.append(ref)
                intents_ok.append(intent_correct)

        t1 = time.perf_counter()
        elapsed = t1 - t0
        calls_per_sec = n_calls / elapsed if elapsed > 0 else np.nan
        mean_sec_per_call = elapsed / max(1, n_calls)

        # Export per-setting CSV
        df_out = pd.DataFrame(per_rows)
        out_csv = os.path.join(args.out_dir, f"{name}.csv")
        df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

        # Aggregation
        bleu_vals = df_out["BLEU4"].dropna().astype(float).tolist() if not df_out.empty else []
        f1_vals = df_out["TokenF1"].dropna().astype(float).tolist() if not df_out.empty else []
        bleu_mean = float(np.mean(bleu_vals)) if bleu_vals else 0.0
        f1_mean = float(np.mean(f1_vals)) if f1_vals else 0.0

        # 95% CI (bootstrap 1000)
        if 'bootstrap_ci' in globals():
            b_low, b_high = bootstrap_ci(bleu_vals, n_boot=1000, alpha=0.05) if bleu_vals else (0.0, 0.0)
            f_low, f_high = bootstrap_ci(f1_vals, n_boot=1000, alpha=0.05) if f1_vals else (0.0, 0.0)
        else:
            b_low = b_high = f_low = f_high = 0.0

        # BERTScore (optional)
        bert_res, bert_err = (None, None)
        if (not args.skip_bertscore) and hyps and refs:
            try:
                bert_res, bert_err = try_bertscore(hyps, refs) if _USE_UTILS_METRICS else (None, "[BERTScore utils missing]")
            except Exception as e:
                bert_err = f"[BERTScore failed] {e}"

        # Failure rates
        n_rows_total = int(len(df_out))
        empty_rate = (n_empty / max(1, n_calls))
        err_rate = (n_errors / max(1, n_calls))

        # Intent accuracy (only for samples with provided gold)
        valid_intent_flags = [x for x in intents_ok if x is not None]
        intent_acc = float(np.mean(valid_intent_flags)) if valid_intent_flags else np.nan

        # Current setting stats
        stats = {
            "setting": name,
            "key": cfg["key"],
            "desc": cfg["desc"],
            "N_rows": n_rows_total,
            "N_questions": n_calls,
            "TotalSeconds": elapsed,
            "MeanSecPerQ": mean_sec_per_call,
            "QPS": calls_per_sec,
            "LLM_calls": n_calls,          # if >0 inference was performed
            "ErrorRate": err_rate,
            "EmptyAnswerRate": empty_rate,
            "BLEU4_mean": bleu_mean,
            "BLEU4_CI_low": b_low,
            "BLEU4_CI_high": b_high,
            "TokenF1_mean": f1_mean,
            "TokenF1_CI_low": f_low,
            "TokenF1_CI_high": f_high,
            "MeanIntentConsistency": float(np.mean(df_out["intent_correct"])) if "intent_correct" in df_out else np.nan,
            "intent_acc": intent_acc,
        }
        if bert_res:
            stats.update({"BERT_P": bert_res["P"], "BERT_R": bert_res["R"], "BERT_F1": bert_res["F1"]})
        if bert_err:
            stats["BERTScore_note"] = bert_err

        # Delta metrics (baseline is B0)
        if cfg["key"] == "B0":
            baseline_stats = stats.copy()
            stats["Delta_BLEU"] = 0.0
            stats["Delta_F1"] = 0.0
        else:
            if baseline_stats is not None:
                stats["Delta_BLEU"] = stats["BLEU4_mean"] - baseline_stats["BLEU4_mean"]
                stats["Delta_F1"] = stats["TokenF1_mean"] - baseline_stats["TokenF1_mean"]
            else:
                stats["Delta_BLEU"] = np.nan
                stats["Delta_F1"] = np.nan

        all_summaries.append(stats)

        # Console brief
        print(f"Setting {name} -> BLEU4={bleu_mean:.4f} [{b_low:.4f},{b_high:.4f}], "
              f"F1={f1_mean:.4f} [{f_low:.4f},{f_high:.4f}], ΔBLEU={stats['Delta_BLEU']:.4f}, ΔF1={stats['Delta_F1']:.4f}, "
              f"Empty={empty_rate:.3f}, Err={err_rate:.3f}, LLM_calls={n_calls}, MeanSecPerQ={mean_sec_per_call:.3f}")

        if bert_res:
            print(f"  BERTScore F1={bert_res['F1']:.4f}")

    # Summary
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    pd.DataFrame(all_summaries).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\nAll done. Summary saved -> {summary_csv}")

if __name__ == "__main__":
    main()
