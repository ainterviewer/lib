import asyncio

from openai import OpenAI

from ainterviewer.settings import settings
from ainterviewer.utils import encode_image


async def get_image():
    server_endpoint = f"{settings.llm.llm_host}:{settings.llm.llm_port}"
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{server_endpoint}/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Single-image input inference
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    image_path = "data/images/cph.jpg"

    chat_response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {
                "role": "user",
                "content": [
                    # NOTE: The prompt formatting with the image token `<image>` is not needed
                    # since the prompt will be processed automatically by the API server.
                    {"type": "text", "text": "Describe what you see in the image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        },
                    },
                ],
            }
        ],
    )
    print("Chat completion output:", chat_response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(get_image())

# from ainterviewer.lpm.clients import visual_chat
#
# if __name__ == "__main__":
#     messages = [
#         {
#             "role": "user",
#             "content": "What is in this picture?",
#             "images": ["data/images/horse.jpg"],
#         }
#     ]
#     message = visual_chat("llava", messages)
