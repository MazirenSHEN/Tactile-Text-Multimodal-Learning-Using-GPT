import ImageBind.data as data
import numpy as np
import openai

# 1. 提取触觉特征
touch = data.load_and_transform_vision_data(["./touch_1/0000016865.jpg"], device="cuda")
touch_embed = touch.cpu().numpy().flatten()  # 假设是 torch tensor
mean = np.mean(touch_embed)
std = np.std(touch_embed)

# 2. 组装 prompt
system_prompt = "You are an expert in tactile sensing and materials science."
user_prompt = f"""
You will be presented with a touch image from an object/surface.
Here are statistics of the tactile embedding for reference:
Mean: {mean:.4f}, Std: {std:.4f}
Can you describe the touch feeling and the texture?
"""

# 3. GPT-4 调用
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
