import csv
import os
import oss2

# 替换为你自己的信息
access_key_id = 'LTAI5tL5C14dFA4HBtCFtfMs'
access_key_secret = 'dqQKrJGuC5xshEU1HlKKOTH0q6Ut81'
bucket_name = 'js1624'
region = 'oss-eu-west-1'  # 你的bucket所在区域
endpoint = f'https://{region}.aliyuncs.com'

tac_dir = "../data/ssvtp/images_tac"
rgb_dir = "../data/ssvtp/images_rgb"
index_csv = "data/ssvtp/new_train.csv"  # 原始 CSV，用于匹配 index
output_csv = "data/ssvtp/image_url_mapping.csv"  # 输出保存路径

# === 初始化 OSS 连接 ===
auth = oss2.Auth(access_key_id, access_key_secret)
bucket = oss2.Bucket(auth, endpoint, bucket_name)

def upload_image(local_path: str, remote_path: str):
    if not os.path.exists(local_path):
        print(f"[跳过] 本地文件不存在: {local_path}")
        return None
    bucket.put_object_from_file(remote_path, local_path)
    url = f"https://{bucket_name}.{region}.aliyuncs.com/{remote_path}"
    print(f"[上传成功] {url}")
    return url

results = []

with open(index_csv, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        index = row["index"]
        tac_filename = f"image_{index}_tac.jpg"
        rgb_filename = f"image_{index}_rgb.jpg"

        tac_path = os.path.join(tac_dir, tac_filename)
        rgb_path = os.path.join(rgb_dir, rgb_filename)

        tac_url = upload_image(tac_path, f"tac/{tac_filename}")
        rgb_url = upload_image(rgb_path, f"rgb/{rgb_filename}")

        results.append({
            "index": index,
            "tac_url": tac_url or "",
            "rgb_url": rgb_url or ""
        })

with open(output_csv, "w", encoding="utf-8", newline="") as fw:
    writer = csv.DictWriter(fw, fieldnames=["index", "tac_url", "rgb_url"])
    writer.writeheader()
    writer.writerows(results)

print(f"✅ 图片 URL 映射已保存到: {output_csv}")
