from openai import OpenAI
import os

PATH = r"C:\Users\ronald\force-di\refactoring\coc_reduktion_result_codestral-2501\full_prompt.txt"

# Read file content
with open(PATH, "r", encoding="utf-8") as f:
    prompt_content = f.read()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")  # set this in your environment
)

completion = client.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
    messages=[
        {"role": "system", "content": prompt_content}
    ],
    stream=True
)

for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")