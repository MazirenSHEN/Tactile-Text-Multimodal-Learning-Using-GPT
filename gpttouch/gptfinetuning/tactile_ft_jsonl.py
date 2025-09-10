#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tactile_ft_jsonl.py — Create tactile-focused examples for second-stage fine-tuning.

Purpose:
This script generates a smaller, controlled dataset (default ≈ 1,500 records;
output `tactile_ft4.jsonl`) intended specifically for a secondary/adapter
fine-tuning step. Its goal is to make the GPT model competent at answering
questions that rely solely on tactile sensor images, without depending on
visual cues. Use this file to narrowly tune the model to reason from tactile
inputs alone.

Key behavior and constraints:
- Emphasizes tactile-only examples (and enforces hard quotas for tactile-only,
  dual-image, and text-only types), producing a curated subset rather than a
  broad dump.
- Starts sampling from a configurable dataset offset (e.g., `start_image_offset`)
  and keeps tight control over how many dual-image and text-only examples are
  emitted, which helps focus learning on tactile reasoning.
- Ideal for a second-stage fine-tune (adapter / low-rate fine-tuning) that
  prioritizes tactile generalization and prevents the model from over-relying on
  visual features learned in the first stage.

Notes:
- Make sure the tactile-only examples are high quality and that captions/labels
  are consistent; noisy tactile labels disproportionately hurt tactile-only
  performance.
- After second-stage fine-tuning, validate on held-out tactile-only queries to
  confirm improved tactile reasoning without regression on multimodal behavior.
"""

import csv
import json
import re
import argparse
import random
from typing import List, Dict

# ----------------------------
# Configuration & templates
# ----------------------------
intent_question_map = {
    "judgement": [
        "Based on the tactile cues, what material is this object likely made of?",
        "Can you infer the material properties from the touch feedback?",
        "What type of substance does this texture suggest?",
        "Is it possible to identify the material class from these tactile features?",
        "What can be deduced about the object's composition through touch?",
    ],
    "property": [
        "What are the tactile features of this sample?",
        "How does the surface feel to the touch?",
        "Describe the touch-based properties of the object.",
        "What tactile sensations are present in this sample?",
        "Can you summarize the touch characteristics of this material?",
    ],
    "comparison": [
        "Compared with stone, how does this tactile texture differ?",
        "What are the tactile differences between this and fabric?",
        "How does this surface feel compared to metal?",
        "Relative to plastic, what tactile traits does this object exhibit?",
        "What distinguishes this touch sensation from rubber?",
    ],
}

TEXT_ONLY_QUESTIONS = [
    "What's the {material} feel like?",
    "Describe the tactile sensation of {material}.",
    "How does the {material} feel to the touch?",
    "What tactile properties does {material} have?",
    "Can you describe the touch-based traits of {material}?"
]

SYSTEM_ROLE = (
    "You are a tactile perception expert. "
    "Answer only based on tactile features. Never describe or infer any visual property. "
    "Provide a concise and efficient response in no more than 2–3 sentences (30 words)."
)

# ----------------------------
# Utility functions
# ----------------------------
def get_user_query(intent: str) -> str:
    intent = intent.strip().lower()
    if intent in intent_question_map:
        return random.choice(intent_question_map[intent])
    return "What tactile information can you derive from this sample?"

def load_synonyms(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_material_mapping(_: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        "rubber-like": ["elastic", "stretchy", "tacky", "rubbery", "springy", "bouncy", "pliable", "grippy", "sticky"],
        "fabric-like": ["soft", "plush", "cloth", "fabric", "woolen", "woven", "fibrous", "fuzzy", "fluffy", "cushioned"],
        "metal-like": ["metallic", "metal", "cold", "smooth hard", "rigid", "shiny", "polished", "glossy"],
        "stone-like": ["grainy", "rough", "coarse", "jagged", "bumpy", "hard", "uneven"],
        "plastic-like": ["plastic", "synthetic", "glossy", "lightweight", "rigid", "smooth"],
        "wood-like": ["wooden", "grain", "fibrous rigid", "warm rigid", "textured"],
        "foam-like": ["spongy", "porous", "absorbent", "soft springy", "cushioned"],
        "glass-like": ["smooth hard", "glossy", "transparent", "polished", "slippery"],
        "paper-like": ["paper", "cardboard", "thin", "plain", "flat"]
    }

def split_labels(captions: str) -> List[str]:
    """Clean, deduplicate, and normalize labels."""
    if not captions:
        return []
    raw = []
    for line in captions.replace("\r", "\n").split("\n"):
        line = re.sub(r"^[0-9A-Za-z]+[\.\):]\s*", "", line.strip(" -•\t"))
        if not line:
            continue
        parts = [p.strip() for p in re.split(r"[;，、]+", line) if p.strip()]
        raw.extend(parts or [line])
    # Normalize + deduplicate while preserving order
    seen, cleaned = set(), []
    for it in raw:
        it = re.sub(r"\s+", " ", it).strip()
        it_low = it.lower()
        if it_low and it_low not in seen:
            seen.add(it_low)
            cleaned.append(it_low)
    return cleaned

def extract_comparison_object(query: str) -> str:
    """Robustly extract a comparison object from the question (supports multiple phrasings)."""
    qs = query.strip().lower()
    patterns = [
        r"between\s+(?:this|the object)\s+and\s+([a-z][a-z\s-]+)[\?\.,;!]*",
        r"between\s+([a-z][a-z\s-]+)\s+and\s+(?:this|the object)[\?\.,;!]*",
        r"compared\s+(?:with|to)\s+([a-z][a-z\s-]+)[\?\.,;!]*",
        r"relative\s+to\s+([a-z][a-z\s-]+)[\?\.,;!]*",
        r"distinguish(?:es)?\s+(?:this|the object|this touch sensation)?\s*from\s+([a-z][a-z\s-]+)[\?\.,;!]*",
        r"与\s*([\u4e00-\u9fffa-z\s-]+)\s*相比",
    ]
    for pat in patterns:
        m = re.search(pat, qs, re.IGNORECASE)
        if m:
            return m.group(1).strip(" .,!;?-").rstrip("s")
    return ""

def guess_material_category(labels: List[str], material_map: Dict[str, List[str]], synonyms: Dict[str, List[str]]) -> str:
    label_text = " ".join(labels).lower()
    scores = {cat: 0 for cat in material_map}
    for cat, keywords in material_map.items():
        for kw in keywords:
            kw_l = kw.lower()
            if kw_l and kw_l in label_text:
                scores[cat] += 1
                continue
            syn_list = synonyms.get(kw_l, []) if kw_l in synonyms else []
            if not syn_list:
                for cano, vals in synonyms.items():
                    if any(kw_l == v.lower() for v in vals):
                        syn_list.extend(vals + [cano])
                        break
            for syn in syn_list:
                if syn and syn.lower() in label_text:
                    scores[cat] += 1
                    break
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else ""

def generate_answer(intent: str, labels: List[str], user_query: str,
                    material_map: Dict[str, List[str]], synonyms: Dict[str, List[str]]) -> str:
    labels_joined = ", ".join(labels) if labels else "neutral, consistent"
    first_label = labels[0] if labels else "neutral"
    last_label = labels[-1] if labels else "consistent"

    if intent == "property":
        return f"The surface feels {labels_joined}, providing a consistent and stable tactile experience."

    elif intent == "comparison":
        ref_obj = extract_comparison_object(user_query)
        uq = user_query.lower()
        if ref_obj:
            if "what distinguishes" in uq or "differences" in uq:
                return f"Key differences from {ref_obj}: {labels_joined}. Emphasis on tactile traits rather than typical {ref_obj} feel."
            if "relative to" in uq:
                return f"Relative to {ref_obj}, it feels {labels_joined}, indicating distinct tactile feedback."
            return f"Compared with {ref_obj}, it feels {labels_joined}, diverging from typical {ref_obj} traits."
        return f"Tactile feel trends toward {first_label} but departs in {last_label}."

    elif intent == "judgement":
        cat = guess_material_category(labels, material_map, synonyms)
        if cat:
            return f"The sample feels {labels_joined}. These cues may indicate a {cat} material."
        else:
            return f"The sample feels {labels_joined}."

    return f"The surface exhibits tactile characteristics: {labels_joined}."

def valid_url(u: str) -> bool:
    return bool(u) and isinstance(u, str) and u.strip().lower().startswith(("http://", "https://"))

# ---------- Text-only sample generation (forward + backward; insert one pair every N images) ----------
def realize_material_name(mat_key: str) -> str:
    """Convert 'rubber-like' -> 'rubber', 'stone-like' -> 'stone'."""
    return mat_key.replace("-like", "")

def make_tactile_phrase(words: List[str]) -> str:
    """Create a short natural phrase from words and control length."""
    chosen = random.sample(words, min(len(words), random.randint(3, 6)))
    return ", ".join(chosen)

def make_text_only_pair(material_map: Dict[str, List[str]]) -> List[Dict]:
    """Return 2 text-only examples: forward (ask material -> answer features) and reverse (features -> answer material)."""
    mat_key = random.choice(list(material_map.keys()))
    words = material_map[mat_key]
    mat_name = realize_material_name(mat_key)

    # Forward
    q1 = random.choice(TEXT_ONLY_QUESTIONS).format(material=mat_name)
    phrase1 = make_tactile_phrase(words)
    a1 = f"It feels {phrase1}, providing a tactile impression typical of {mat_name}."
    rec1 = {
        "messages": [
            {"role": "system", "content": SYSTEM_ROLE},
            {"role": "user", "content": q1},
            {"role": "assistant", "content": a1}
        ]
    }

    # Reverse
    phrase2 = make_tactile_phrase(words)
    q2 = f"It feels {phrase2}. What is the likely material?"
    a2 = f"These tactile cues suggest a {mat_key} material."
    rec2 = {
        "messages": [
            {"role": "system", "content": SYSTEM_ROLE},
            {"role": "user", "content": q2},
            {"role": "assistant", "content": a2}
        ]
    }
    return [rec1, rec2]

# ----------------------------
# Main flow
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/ssvtp/new_train.csv",
                    help="Caption CSV with columns: index, caption")
    ap.add_argument("--image-url-csv", default="data/ssvtp/image_url_mapping.csv",
                    help="Image URL CSV with columns: index, tac_url, rgb_url")
    ap.add_argument("--synonyms", default="data/ssvtp/synonyms.json")
    ap.add_argument("--out", default="data/ssvtp/tactile_ft4.jsonl")
    ap.add_argument("--limit", type=int, default=1500,
                    help="Max number of QA samples to generate (total).")
    ap.add_argument("--image-limit", type=int, default=None,
                    help="Max number of unique images (indices) to use.")
    ap.add_argument("--intents", nargs="+",
                    default=["judgement", "comparison", "property"],
                    help="Which intents to generate for image-based samples.")
    ap.add_argument("--text-only-every", type=int, default=5,
                    help="Insert text-only (forward+backward) after every N images. Default=5.")
    ap.add_argument("--text-only-pairs", type=int, default=1,
                    help="How many text-only pairs to insert each time. Default=1 (i.e., 2 samples).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = ap.parse_args()

    random.seed(args.seed)

    # Load synonyms and material mapping
    synonyms = load_synonyms(args.synonyms)
    material_map = build_material_mapping(synonyms)

    # Load image URL mapping
    url_map = {}
    with open(args.image_url_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = (row.get("index") or "").strip()
            if not idx:
                continue
            tac_url = (row.get("tac_url") or "").strip()
            rgb_url = (row.get("rgb_url") or "").strip()
            url_map[idx] = {"tac_url": tac_url, "rgb_url": rgb_url}

    # Load captions
    rows = []
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Generate JSONL
    sample_count = 0
    image_used = 0
    with open(args.out, "w", encoding="utf-8") as fw:
        # ↓↓↓ Add image usage start offset (start from image #900)
        start_image_offset = 900
        rows = rows[start_image_offset:]

        # ↓↓↓ Sample quotas: 400 tactile-only + 35 dual-image + 100 text-only
        max_tactile_only = 400
        max_dual_image = 35
        max_text_only = 100
        used_tactile_only = 0
        used_dual_image = 0
        used_text_only = 0

        sample_count = 0
        with open(args.out, "w", encoding="utf-8") as fw:
            for row in rows:
                if (used_tactile_only >= max_tactile_only and
                        used_dual_image >= max_dual_image and
                        used_text_only >= max_text_only):
                    break

                idx = (row.get("index") or "").strip()
                captions = (row.get("caption") or "").strip()
                if not idx or idx not in url_map:
                    continue

                labels = split_labels(captions)
                if not labels:
                    continue

                tac_url = url_map[idx]["tac_url"]
                rgb_url = url_map[idx]["rgb_url"]

                if not valid_url(tac_url):
                    continue  # Skip if there is no tactile image

                # Prefer tactile-only examples
                image_content = [{"type": "image_url", "image_url": {"url": tac_url, "detail": "low"}}]

                # Dual-image condition
                dual_mode = valid_url(rgb_url) and used_dual_image < max_dual_image
                if dual_mode:
                    image_content.append({"type": "image_url", "image_url": {"url": rgb_url, "detail": "low"}})

                # Control sample generation
                for intent in args.intents:
                    if sample_count >= args.limit:
                        break
                    user_query = get_user_query(intent)
                    answer = generate_answer(intent, labels, user_query, material_map, synonyms)
                    if not answer:
                        continue

                    content = [{"type": "text", "text": user_query}] + image_content
                    rec = {
                        "messages": [
                            {"role": "system", "content": SYSTEM_ROLE},
                            {"role": "user", "content": content},
                            {"role": "assistant", "content": answer}
                        ]
                    }
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    sample_count += 1

                # Update counters
                if dual_mode:
                    used_dual_image += 1
                elif used_tactile_only < max_tactile_only:
                    used_tactile_only += 1

                # After every 10 image-based usages (tactile-only or dual-image), insert text-only pairs (if under quota)
                total_images_used = used_tactile_only + used_dual_image
                if (total_images_used % 10 == 0 and used_text_only < max_text_only):
                    pair = make_text_only_pair(material_map)
                    for rec in pair:
                        if used_text_only >= max_text_only:
                            break
                        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        used_text_only += 1
                        sample_count += 1

    # Rough cost estimate (1 epoch): conservative upper bound assuming 250 tokens per sample
    est_tokens_upper = sample_count * 250
    est_cost_upper = est_tokens_upper / 1_000_000 * 25  # $25 / 1M tokens (GPT-4o training)
    print(f"Wrote {sample_count} records to {args.out}")
    print(f"[Upper bound] Estimated tokens (1 epoch): {est_tokens_upper:,}")
    print(f"[Upper bound] Estimated training cost (1 epoch @ $25/M): ${est_cost_upper:.2f}")

if __name__ == "__main__":
    main()
