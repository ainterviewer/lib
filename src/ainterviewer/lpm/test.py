import os

from any_llm import completion
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ALEX_API_KEY")

print(api_key)

response = completion(
    model="openai:qwen3.5-397b",
    api_key=api_key,
    api_base="https://inference.alexandra.dk/v1",
    messages=[
        {"role": "system", "content": "Du er en hjælpsommelig assistent"},
        {
            "role": "user",
            "content": "svar med 20 ord. Hvordan går det?",
        },
    ],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

print(response)
