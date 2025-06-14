import openai

openai.api_key = "your-api-key"

response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "You are an expert in tactile understanding."},
    {"role": "user", "content": "This is a touch image of an object... [embedding or description]. Can you describe the texture and feeling?"}
  ],
  temperature=0.7,
)

print(response['choices'][0]['message']['content'])
