import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# SET YOUR NVIDIA API KEY
# ============================================================

API_KEY = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY
)

# ============================================================
# LIST AVAILABLE MODELS
# ============================================================

models = client.models.list()

print("\nAvailable Models:\n")

model_names = []

for i, model in enumerate(models.data):
    print(f"{i + 1}. {model.id}")
    model_names.append(model.id)

# ============================================================
# CHOOSE MODEL
# ============================================================

choice = int(input("\nEnter model number: "))
selected_model = model_names[choice - 1]

print(f"\nSelected Model: {selected_model}")

# ============================================================
# USER PROMPT
# ============================================================

prompt = input("\nEnter your prompt:\n\n")

# ============================================================
# GENERATE RESPONSE
# ============================================================

response = client.chat.completions.create(
    model=selected_model,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.7,
    # max_tokens=1024
)

output = response.choices[0].message.content

# ============================================================
# SAVE OUTPUT
# ============================================================

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("\nOutput saved to output.txt")