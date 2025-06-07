# Chatbot Service

This directory contains a small REST chatbot built with FastAPI and OpenAI. The service keeps conversation history for up to 15 minutes so short sessions can be maintained before starting over.

## Files

- `agent_service.py` – exposes a `/chat` endpoint that sends accumulated messages to OpenAI and stores replies.
- `Dockerfile` – container definition for running the service.

## Usage

Install dependencies locally:

```bash
pip install -r requirements.txt
```

Run the service on port 8000:

```bash
uvicorn agent_service:app --port 8000
```

Set the `OPENAI_API_KEY` environment variable before launching the server.

### Docker

Build and start a container:

```bash
docker build -t chatbot .
docker run -p 8000:8000 -e OPENAI_API_KEY=YOUR_KEY chatbot
```

Send a message:

```bash
curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{"message": "Hola"}'
```

A new conversation begins automatically if 15 minutes pass since the first user message.
