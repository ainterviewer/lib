# LLM interfaces

The choice of Large Language Model is up to the user. GPT-4 provide good results, but we deploy and encourage the use of locally hosted models.

## Model Evaluations

### Qwen 1.5

#### 32b

This model seems too verbose.

Also randomly generates Chinese characters.

> AInterviewer: Do you recall any specific event or piece of information that sparked your interest in the green transition?
> user: I read about it in the newspapers.
> AInterviewer: Given your interest was piqued by reading about the green transition in newspapers, what stood out to you in those articles that made you want to learn more or sparked your curiosity?
> user: What stood out to me was especially the sense of urgency to which we must respond to the climate change.
> AInterviewer: It's great to hear that sense of urgency触动了你。Did any particular examples or anecdotes in those articles help you relate to the issue on a personal level, or did it mainly come from a global perspective?

## Mistral

https://docs.mistral.ai/quickstart#install-vllm
https://docs.mistral.ai/usage/how-to-use

Create a new environment (!Not the same one as ainterviewer, since they have dependency conflicts) and install the following packages.

### Setup

```console
pip install vllm
pip install fschat
pip install accelerate
```

Authenticate with huggingface cli

```console
huggingface-cli login
```

Run an instance of the model which you can interact with via a CLI.

This assumes that you are connecting through port 8002

```console
python -u -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8667 --model mistralai/Mistral-7B-Instruct-v0.1
```

To use a specific GPU device;

```console
CUDA_VISIBLE_DEVICES=1 python -u -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8002 --model mistralai/Mistral-7B-Instruct-v0.1
```

### Usage

```python
import openai

openai.api_base = "http://0.0.0.0:8667/v1"
openai.api_key = "none"

messages = [
    {
        "role": "user",
        "content": "What is the bash command to list all files in a folder and sort them by last modification?",
    }
]

chat_completion = openai.ChatCompletion.create(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=1,
    max_tokens=1024,
    messages=messages,
)

print(chat_completion)
```

## Qwen

https://github.com/QwenLM/Qwen#vllm

You can use same Python environment and setup as above, just change model to `"Qwen/Qwen-7B-Chat"`

## OpenChat

https://github.com/imoneoi/openchat
https://github.com/imoneoi/openchat#-deploying-api-server

```bash
pip install ochat
```

To serve the API:

```bash
python -m ochat.serving.openai_api_server --model openchat/openchat_3.5
```

And then change model to `"openchat_3.5"` in calls.

## ollama & LiteLLM

In order to get self-served quantized models to work, we suggest that you use a combination of ollama and LiteLLM.

Download ollama from source:

```bash
curl -L https://ollama.ai/download/ollama-linux-amd64 -o ~/bin/ollama
chmod +x ~/bin/ollama
```

Start the ollama server

```bash
OLLAMA_MODELS=/scratch/hwl272/.cache/ollama/ ollama serve # Specify the host and port with OLLAMA_HOST=0.0.0.0:8667
```

Pull the desired model, ie. mixtral:

```bash
ollama pull mixtral
```

Create an environment and install litellm

```bash
pip install "litellm[proxy]"
```

Start the litellm proxy server

```bash
litellm --model ollama/mixtral --port 8667 --api_base "http://localhost:11434"
```
