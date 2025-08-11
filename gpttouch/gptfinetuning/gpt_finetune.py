import openai
from openai import OpenAI
import time

# 初始化 OpenAI 客户端（推荐使用新版 SDK）
client = OpenAI(api_key="sk-")  # ← 换成你自己的 API Key

# Step 1: 上传文件
file_path = "../data/ssvtp/tactile_ft.jsonl"
print("🔼 Uploading file...")

upload_response = client.files.create(
    file=open(file_path, "rb"),
    purpose="fine-tune"
)

file_id = upload_response.id
print(f"✅ File uploaded. File ID: {file_id}")

# Step 2: 创建微调作业
print("🚀 Creating fine-tuning job...")

fine_tune_job = client.fine_tuning.jobs.create(
    training_file=file_id,
    model="gpt-4o-2024-08-06"
)

job_id = fine_tune_job.id
print(f"📋 Fine-tuning job created. Job ID: {job_id}")

# Step 3: 可选，等待训练完成（轮询）
print("⏳ Waiting for fine-tuning job to complete (may take several minutes)...")

while True:
    job_status = client.fine_tuning.jobs.retrieve(job_id)
    status = job_status.status
    print(f"⏱️ Current status: {status}")
    if status in ["succeeded", "failed", "cancelled"]:
        break
    time.sleep(15)

# Step 4: 输出最终模型名
if status == "succeeded":
    print(f"✅ Fine-tuning complete! Model: {job_status.fine_tuned_model}")
else:
    print(f"❌ Fine-tuning did not succeed. Final status: {status}")
