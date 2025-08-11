import base64
import os
from openai import OpenAI

# ✅ 替换为你的 OpenAI API Key
client = OpenAI(api_key="sk-proj-")  # 推荐从环境变量读取

def encode_image_to_base64(image_path: str) -> str:
    """将本地图片编码为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_multimodal_prompt(image_path: str, user_text: str) -> list:
    """构建包含图片和文字的用户消息"""
    base64_image = encode_image_to_base64(image_path)
    return [
        {"type": "text", "text": user_text},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "low"  # 可设为 "low" 以降低 token 消耗
            }
        }
    ]

def ask_gpt4_with_image(image_path: str, question: str):
    """调用 GPT-4o 视觉模型进行图片理解"""
    messages = [
        {"role": "system", "content": "You are a tactile assistant that describes GelSight images."},
        {"role": "user", "content": build_multimodal_prompt(image_path, question)}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",  # 或 "gpt-4-vision-preview"
        messages=messages,
        max_tokens=256,
        temperature=0.7,
    )

    print("🤖 Assistant response:\n")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    # ✅ 替换为你的图片路径
    image_path = "../data/ssvtp/images_rgb/image_123_rgb.jpg"
    question = "请根据这张图像描述物体表面纹理与硬度。"

    ask_gpt4_with_image(image_path, question)
