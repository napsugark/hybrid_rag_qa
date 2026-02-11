#!/bin/bash
# Custom entrypoint for Ollama: starts the server, then pulls the model if missing.

# Start Ollama server in the background
ollama serve &

# Wait for the server to be ready
echo "Waiting for Ollama server to start..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
done
echo "Ollama server is ready."

# Pull the model if not already present
MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
if ollama list | grep -q "^${MODEL}"; then
    echo "Model '${MODEL}' already available."
else
    echo "Pulling model '${MODEL}'... (this may take a while on first run)"
    ollama pull "${MODEL}"
    echo "Model '${MODEL}' pulled successfully."
fi

# Bring the background server back to the foreground
wait
