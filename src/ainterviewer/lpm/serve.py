from argparse import ArgumentParser
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

from ainterviewer.settings import settings

arg_parser = ArgumentParser()
arg_parser.add_argument("--model", type=str, default="allenai/OLMo-7B-Instruct-hf")
arg_parser.add_argument("--max_new_tokens", type=int, default=400)
arg_parser.add_argument("--device", type=str, default="cuda")
arg_parser.add_argument("--port", type=int, default=settings.llm.llm_port)

args = arg_parser.parse_args()

app = FastAPI()

# Initialize the pipeline
pipe = pipeline(
    "text-generation",
    model=args.model,
    device=args.device,
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: int = args.max_new_tokens
    temerature: float = 0.7
    do_sample: bool = True


@app.post("/generate")
async def generate_text(request: ChatRequest):
    try:
        # Convert the incoming messages to the format expected by the pipeline
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # Generate the response
        response = pipe(
            formatted_messages,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temerature,
            do_sample=request.do_sample,
        )

        # Extract the generated text from the response
        generated_text = response[0]["generated_text"]

        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("Starting server with the following configuration:")
    print(f"Model: {args.model}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Device: {args.device}")
    print(f"Port: {args.port}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
