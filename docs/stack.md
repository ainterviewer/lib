# aInterviewer software stack

Currently, the project consists of 3 different pieces.

**1\. The ainterviewer package**  
This is our python package that is the workhorse behind conducting the semistructured interviews using LLMs.

**2\. The LLMs**  
The LLMs that are used by the ainterviewer, can either be from a provider, eg. OpenAI, or self-hosted, ie. via `vllm`.

**3\. Interface**  
The interface facing the users. Currently a simple HTML/js frontend is used with a python fastapi backend.

Below is a rundown of the required packages.

## ainterviewer package:

Python packages:

```text
openai
websocket-client
pydantic>=2.4.2
python-dotenv
PyYAML
```

## LLMs

```text
vllm
fschat
accelerate
huggingface_hub[cli]
```

This infrastucture is used to serve LLMs downloaded from the huggingface hub, eg.

https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2  
https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1


## Interface
### Backend

Python packages:

```text
uvicorn[standard]
fastapi
supabase
beautifulsoup4
openai
python-dotenv
starlette
pydantic>=2.4.2
```

and the ainterviewer package.

### Frontend
Currently runs on vanilla HTML/js with WebSocket

