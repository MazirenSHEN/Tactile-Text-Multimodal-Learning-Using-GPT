import torch
import ImageBind.data as data
import numpy as np
import openai

# 1. 用 ImageBind 预处理图片，得到张量
touch_tensor = data.load_and_transform_vision_data(
    ["your_touch_image.jpg"], device="cuda"
)

# 2. 用 ImageBind backbone 得到 embedding
# 假设你有 imagebind_model 已经加载好
# 例如 imagebind_model = ImageBindModel.load_pretrained(...).to("cuda")
with torch.no_grad():
    embedding = imagebind_model(touch_tensor)  # 输出一般 shape=(1, embedding_dim)
embedding = embedding.cpu().numpy().flatten()

# 3. 统计 embedding
mean = np.mean(embedding)
std = np.std(embedding)

# 4. 拼 prompt
system_prompt = "You are an expert in tactile sensing and materials science."
user_prompt = f"""
Below is the tactile embedding of an object, extracted by a multimodal AI model.
Mean: {mean:.4f}, Std: {std:.4f}
What does this surface feel like? Please describe in detail.
"""

# 5. GPT-4 调用
openai.api_key = "YOUR_OPENAI_API_KEY"
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.7,
)
print(response['choices'][0]['message']['content'])
