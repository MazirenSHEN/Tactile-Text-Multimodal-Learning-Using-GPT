import os
import re
import math
import random
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np

# -----------------------------
# Path / ID helpers
# -----------------------------
ID_PAT = re.compile(r"image_(\d+)(?:_tac)?\.(?:jpg|jpeg|png|bmp|webp)$", re.IGNORECASE)

def parse_tac_id_from_path(path: str) -> Optional[int]:
    """
    Extract integer ID from a tactile image path. Supports patterns like:
      .../images_tac/image_473_tac.jpg
      .../images_tac/image_473.jpg
      .../image_473_tac.png
    Returns None if it cannot be parsed.
    """
    if not path:
        return None
    fname = os.path.basename(path)
    m = ID_PAT.search(fname)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # Fallback: search anywhere in the filename for image_<digits>
    m2 = re.search(r"image_(\d+)", fname, re.IGNORECASE)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            pass
    # Last resort: take the last group of digits in the name
    m3 = re.findall(r"(\d+)", fname)
    if m3:
        try:
            return int(m3[-1])
        except Exception:
            pass
    return None

def compute_recall_at_k(ranks: List[int], k: int) -> float:
    """ranks: 1-based rank if found, or 0 if not found."""
    hits = sum(1 for r in ranks if 1 <= r <= k)
    return hits / max(1, len(ranks))


def compute_map(ranks: List[int]) -> float:
    """
    mAP for single relevant item per query:
    AP = 1/rank if found else 0. mAP is mean of APs.
    """
    aps = [(1.0 / r) if r > 0 else 0.0 for r in ranks]
    return float(np.mean(aps)) if len(aps) > 0 else 0.0


def compute_median_rank(ranks: List[int]) -> float:
    vals = [r if r > 0 else float('inf') for r in ranks]
    vals_sorted = sorted(vals)
    mid = len(vals_sorted) // 2
    if len(vals_sorted) % 2 == 1:
        return float(vals_sorted[mid])
    return (vals_sorted[mid - 1] + vals_sorted[mid]) / 2.0


def compute_mean_rank(ranks: List[int]) -> float:
    vals = [r if r > 0 else float('inf') for r in ranks]
    return float(np.mean(vals)) if len(vals) > 0 else float('inf')


# -----------------------------
# Bootstrap CI
# -----------------------------
def bootstrap_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """
    Returns (low, high) percentile CI for the mean of 'values' using bootstrap.
    """
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    boots = []
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boots.append(float(np.mean(sample)))
    low = float(np.percentile(boots, 100 * (alpha / 2)))
    high = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (low, high)


def bootstrap_ci_metric(
    ranks: List[int],
    metric_fn: Callable[[List[int]], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap CI for any metric that maps ranks->[0,1] (like R@1, R@5, mAP).
    We'll resample the list of ranks with replacement and recompute the metric.
    """
    if len(ranks) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(ranks)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [ranks[i] for i in idx]
        boots.append(metric_fn(sample))
    low = float(np.percentile(boots, 100 * (alpha / 2)))
    high = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (low, high)


# -----------------------------
# Text metrics (BLEU-4, token F1, optional BERTScore)
# -----------------------------
def tokenize_simple(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())

def ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts = {}
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i+n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts

def bleu4(hyp: str, ref: str, smooth: bool = True) -> float:
    """
    Simple BLEU-4 implementation with uniform weights and brevity penalty.
    """
    hyp_tokens = tokenize_simple(hyp)
    ref_tokens = tokenize_simple(ref)
    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    weights = [0.25, 0.25, 0.25, 0.25]
    precisions = []

    for n in range(1, 5):
        hyp_counts = ngram_counts(hyp_tokens, n)
        ref_counts = ngram_counts(ref_tokens, n)
        overlap = 0
        total = max(1, sum(hyp_counts.values()))
        for ng, c in hyp_counts.items():
            overlap += min(c, ref_counts.get(ng, 0))
        p_n = overlap / total
        if smooth and p_n == 0.0:
            # Chen & Cherry smoothing1
            p_n = 1.0 / (total * 2.0)
        precisions.append(p_n)

    # geometric mean
    s = sum(w * math.log(p) for w, p in zip(weights, precisions))
    geo_mean = math.exp(s)

    # brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - (ref_len / max(1, hyp_len)))

    return bp * geo_mean


def token_f1(hyp: str, ref: str) -> float:
    """
    Token-level F1 (micro): precision/recall on set of tokens.
    """
    hyp_set = set(tokenize_simple(hyp))
    ref_set = set(tokenize_simple(ref))
    if len(hyp_set) == 0 and len(ref_set) == 0:
        return 1.0
    if len(hyp_set) == 0 or len(ref_set) == 0:
        return 0.0
    tp = len(hyp_set & ref_set)
    prec = tp / len(hyp_set) if len(hyp_set) > 0 else 0.0
    rec = tp / len(ref_set) if len(ref_set) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def try_bertscore(hyps: List[str], refs: List[str], model_type: str = "microsoft/deberta-xlarge-mnli"):
    """
    Optional BERTScore if bert_score is installed.
    Returns dict with keys 'P','R','F1' (floats).
    """
    try:
        from bert_score import score as bert_score
    except Exception as e:
        return None, f"[BERTScore unavailable] {e}"

    try:
        P, R, F1 = bert_score(hyps, refs, lang="en", model_type=model_type)
        return {
            "P": float(P.mean().item()),
            "R": float(R.mean().item()),
            "F1": float(F1.mean().item()),
        }, None
    except Exception as e:
        return None, f"[BERTScore failed] {e}"


# -----------------------------
# intent_consistency_score (lazy SBERT load + fallback heuristics)
# Append this block to the end of ablation_utils.py
# -----------------------------
_INTENT_SBERT = None
_INTENT_SBERT_UTIL = None

def _load_intent_sbert():
    """
    Lazily load sentence-transformers model and util.
    Returns tuple (model, util) on success, or (None, None) on failure.
    """
    global _INTENT_SBERT, _INTENT_SBERT_UTIL
    if _INTENT_SBERT is None:
        try:
            from sentence_transformers import SentenceTransformer, util
            _INTENT_SBERT = SentenceTransformer("all-mpnet-base-v2")
            _INTENT_SBERT_UTIL = util
        except Exception:
            _INTENT_SBERT = False  # mark as attempted but unavailable
            _INTENT_SBERT_UTIL = None
    return (_INTENT_SBERT if _INTENT_SBERT not in (None, False) else None, _INTENT_SBERT_UTIL)

def build_intent_prototypes():
    """
    Intent prototypes — short canonical texts describing each intent.
    """
    return {
        "property": "Describe tactile properties like texture, hardness, roughness, compliance, temperature.",
        "comparison": "Compare this sample to another by describing differences or similarities in tactile qualities.",
        "judgement": "Make a judgement or identify the material from tactile cues; state uncertainty if needed.",
        "other": "Answer using tactile reasoning.",
    }

def intent_consistency_score(answer: str, intent: str) -> float:
    """
    Score in [0,1] measuring how well 'answer' matches the 'intent'.
    - If sentence-transformers available, use cosine similarity between answer and intent prototype.
    - Else, fall back to keyword heuristics.
    Keeps lightweight import behavior (lazy load on first call).
    """
    if not intent or not isinstance(answer, str):
        return 0.0
    intent_key = str(intent).strip().lower()
    prototypes = build_intent_prototypes()
    proto = prototypes.get(intent_key, prototypes["other"])

    model, util = _load_intent_sbert()
    if model is not None and util is not None:
        try:
            emb_a = model.encode(answer, convert_to_tensor=True, normalize_embeddings=True)
            emb_p = model.encode(proto, convert_to_tensor=True, normalize_embeddings=True)
            sim = float(util.pytorch_cos_sim(emb_a, emb_p).item())  # -1..1
            return max(0.0, min(1.0, (sim + 1.0) / 2.0))
        except Exception:
            # fallthrough to heuristic
            pass

    # fallback heuristics (simple, deterministic)
    a = (answer or "").lower()
    if intent_key == "comparison":
        kws = [" than ", "compare", "compared", " vs ", "versus", "more ", "less ", "similar", "difference", "different"]
        return 1.0 if any(k in a for k in kws) else 0.0
    if intent_key == "property":
        kws = ["texture", "rough", "smooth", "soft", "hard", "sticky", "slippery", "firm", "compliant"]
        return 1.0 if any(k in a for k in kws) else 0.0
    if intent_key == "judgement":
        kws = ["probably", "likely", "material", "metal", "wood", "plastic", "rubber", "glass", "i think", "it is"]
        return 1.0 if any(k in a for k in kws) else 0.0
    return 0.0