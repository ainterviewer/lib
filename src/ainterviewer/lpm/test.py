from openai import OpenAI

base_url = "http://13.50.163.140:8880/v1"

client = OpenAI(base_url=base_url, api_key="")

completion = client.chat.completions.create(
    model="google/gemma-3-27b-it",
    messages=[
        {"role": "system", "content": "You are a helpfull assistant."},
        {
            "role": "user",
            "content": "respond in 20 words. who are you?",
        },
    ],
)

print(completion.choices[0].message.content)


# response = completion(
#     model="hosted_vllm/google/gemma-3-27b-it",
#     messages=[{"content": "respond in 20 words. who are you?", "role": "user"}],
#     api_base=base_url,
#     n=2,
# )
