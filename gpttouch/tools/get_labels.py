import json
from glob import glob

def extract_keywords_to_json(label_pattern="../data/ssvtp/text/labels_*.txt", output_file="tactile_keywords.json"):
    all_keywords = []

    for file in glob(label_pattern):
        with open(file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            for line in lines:
                all_keywords.append({"keywords": line})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_keywords, f, ensure_ascii=False, indent=2)

    print(f"✅ 提取完成，已保存 {len(all_keywords)} 条记录到 {output_file}")

# === 执行 ===
if __name__ == "__main__":


    #label_files = glob("../data/ssvtp/text/labels_*.txt")
    #print("匹配到的文件有：", label_files)

    extract_keywords_to_json()
