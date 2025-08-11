import json
import random
from pathlib import Path

# 载入关键词与问题
with open("/mnt/data/c36596cd-22ac-4db3-a424-b3b7fa9d65c0.json", "r", encoding="utf-8") as f:
    keywords_data = json.load(f)

with open("/mnt/data/ac81245e-0636-4255-9dc7-4b3096ccce64.json", "r", encoding="utf-8") as f:
    questions_data = json.load(f)

questions = [q["question"] for q in questions_data]


def generate_answer(keywords: str) -> str:
    templates = [
        f"It feels {keywords}, with a unique sensory character.",
        f"A distinctly {keywords} texture — subtle yet expressive.",
        f"This surface combines sensations of {keywords}.",
        f"A {keywords} feel, evocative and tactile.",
        f"The material texture is {keywords} in nature.",
        f"A texture that is mostly {keywords}, inviting to touch.",
        f"This evokes a {keywords} tactile quality.",
        f"You can feel the {keywords} features clearly.",
        f"It delivers a {keywords} impression to the touch.",
        f"Touching this, you'd notice it's {keywords}."
    ]
    return random.choice(templates)

# 构造输出
output = []
q_idx = 0
for entry in keywords_data:
    keywords = entry.get("keywords", "").strip()
    if not keywords:
        continue

    question = questions[q_idx % len(questions)]
    answer = generate_answer(keywords)

    approx_token_count = len((question + answer).split()) * 1.3
    if approx_token_count > 100:
        continue

    output.append({
        "messages": [
            {"role": "system", "content": "You are a tactile expert assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    })

    q_idx += 1

# 保存为 JSONL 文件
output_path = Path("/mnt/data/tactile_finetune_natural.jsonl")
with output_path.open("w", encoding="utf-8") as f:
    for item in output:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 已生成 {len(output)} 条微调样本 → {output_path.name}")
