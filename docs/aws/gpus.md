# LLM Inference on EC2 GPU instances

## Ollama

### systemd services

Set base ollama service env vars with `systemctl edit`, so any changes to the
service file are not overwritten on update.

```bash
sudo systemctl edit ollama.service

```

```systemd
[Service]
Environment="OLLAMA_HOST=0.0.0.0:8880"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_CONTEXT_LENGTH=4096"
```

Add a new model loader service, runs after ollama service to preload model the required model

`ollama-model-loader.service`

```systemd
[Unit]
Description=Ollama Model Loader
After=ollama.service
Requires=ollama.service

[Service]
Type=oneshot
ExecStart=/home/ec2-user/load-ollama-model.sh
User=ec2-user
Environment="OLLAMA_HOST=0.0.0.0:8880"

[Install]
WantedBy=multi-user.target
```

### Extra

`load-ollama-model.sh`

```bash
#!/bin/bash
# Script to load Ollama model
# You can modify this script or the config file remotely as needed

# Set variables
CONFIG_FILE="/home/ec2-user/ollama-model-config.json"
MAX_RETRIES=5
RETRY_DELAY=2

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file $CONFIG_FILE not found!"
  exit 1
fi

echo "Starting Ollama model loader using config from $CONFIG_FILE"

# Try to load model with retries
count=0
while [ $count -lt $MAX_RETRIES ]; do
  echo "Attempt $((count+1)) to load model..."

  status_code=$(curl -s "http://$OLLAMA_HOST/api/chat" \
    -d @"$CONFIG_FILE" \
    -H "Content-Type: application/json" \
    -o /dev/null -w "%{http_code}")

  if [ "$status_code" = "200" ]; then
    echo "Successfully loaded model"
    exit 0
  else
    echo "Server returned $status_code, retrying in $RETRY_DELAY seconds..."
    sleep $RETRY_DELAY
    count=$((count+1))
  fi
done

echo "Failed to load model after $MAX_RETRIES attempts"
exit 1
```

`ollama-model-config.json`

```json
{
  "model": "gemma3:27b",
  "options": {
    "num_ctx": 4096
  }
}
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable ollama-model-loader.service
```

## vLLM

```bash
uv tool install vllm \
  --with bitsandbytes \
  --with flashinfer-python --extra-index-url https://flashinfer.ai/whl/cu124/torch2.6/
```

```systemd
[Unit]
Description=vLLM Inference Server
After=network.target

[Service]
Type=simple
User=ec2-user
Environment="PATH=/home/ec2-user/.local/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/etc/vllm/environment
WorkingDirectory=/home/ec2-user
ExecStart=/home/ec2-user/launch.sh
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```
