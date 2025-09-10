import csv
import json
import random

# Modify to your input path
test_csv = "data/ssvtp/test.csv"

# Output file paths
output_question = "data/test_questions.jsonl"
output_answer = "data/test_answers.jsonl"
output_qa = "data/test_qa_pairs.jsonl"

# Question templates
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

# Material category keywords
material_map = {
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

# Utility functions
def split_labels(captions: str):
    raw = []
    for line in captions.replace("\r", "\n").split("\n"):
        line = line.strip(" -•\t")
        if not line:
            continue
        parts = [p.strip() for p in line.split(";") if p.strip()]
        raw.extend(parts or [line])
    seen, cleaned = set(), []
    for it in raw:
        it = it.lower().strip()
        if it and it not in seen:
            seen.add(it)
            cleaned.append(it)
    return cleaned

def guess_material(labels):
    label_text = " ".join(labels)
    scores = {cat: 0 for cat in material_map}
    for cat, keywords in material_map.items():
        for kw in keywords:
            if kw in label_text:
                scores[cat] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else ""

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

# Read test set and construct samples
questions, answers, qa_pairs = [], [], []

with open(test_csv, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        idx = row.get("index", "").strip()
        caption = row.get("caption", "").strip()
        labels = split_labels(caption)
        if not labels:
            continue

        labels_joined = ", ".join(labels)
        mat_cat = guess_material(labels)

        for intent, qs in intent_question_map.items():
            q = random.choice(qs)
            if intent == "judgement":
                if mat_cat:
                    a = f"The sample feels {labels_joined}. These cues may indicate a {mat_cat} material."
                else:
                    a = f"The sample feels {labels_joined}."
            elif intent == "property":
                a = f"The surface feels {labels_joined}, providing a consistent and stable tactile experience."
            elif intent == "comparison":
                a = f"Compared with stone, it feels {labels_joined}. It lacks the typical stone traits."
            elif intent == "comparison":

                uq = q.lower()

                if "what distinguishes" in uq or "differences" in uq:
                    a = f"Key differences from stone: {labels_joined}. Emphasis on tactile traits rather than typical stone feel."
                if "relative to" in uq:
                    a = f"Relative to stone, it feels {labels_joined}, indicating distinct tactile feedback."
                a = f"Compared with stone, it feels {labels_joined}, diverging from typical stone traits."

            else:
                a = f"Tactile characteristics: {labels_joined}."

            questions.append({"index": idx, "intent": intent, "question": q})
            answers.append({"index": idx, "intent": intent, "answer": a})
            qa_pairs.append({"index": idx, "intent": intent, "question": q, "answer": a})

# Write JSONL files
write_jsonl(output_question, questions)
write_jsonl(output_answer, answers)
write_jsonl(output_qa, qa_pairs)

print("✅ Test set construction complete:")
print(f"- Question file: {output_question}")
print(f"- Answer file:   {output_answer}")
print(f"- QA pairs file: {output_qa}")
