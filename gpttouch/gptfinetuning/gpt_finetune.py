import os
import time
from openai import OpenAI

# Initialize OpenAI client (recommended: set OPENAI_API_KEY in environment)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable before running this script.")
client = OpenAI(api_key=api_key)

# Step 1: upload file
file_path = "../data/ssvtp/tactile_ft.jsonl"
print("🔼 Uploading file...")

upload_response = client.files.create(
    file=open(file_path, "rb"),
    purpose="fine-tune"
)

file_id = upload_response.id
print(f"✅ File uploaded. File ID: {file_id}")

# Step 2: create fine-tuning job
print("🚀 Creating fine-tuning job...")

fine_tune_job = client.fine_tuning.jobs.create(
    training_file=file_id,
    model="gpt-4o-2024-08-06"  # change if you want a different base model
)

job_id = fine_tune_job.id
print(f"📋 Fine-tuning job created. Job ID: {job_id}")

# Step 3: optional — poll for job status until completion
print("⏳ Waiting for fine-tuning job to complete (may take several minutes)...")

while True:
    job_status = client.fine_tuning.jobs.retrieve(job_id)
    status = job_status.status
    print(f"⏱️ Current status: {status}")
    if status in ["succeeded", "failed", "cancelled"]:
        break
    time.sleep(15)

# Step 4: print final model name if succeeded
if status == "succeeded":
    print(f"✅ Fine-tuning complete! Model: {job_status.fine_tuned_model}")
else:
    print(f"❌ Fine-tuning did not succeed. Final status: {status}")
