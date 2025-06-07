# Multi-Agent Chat

This example demonstrates how to run multiple chat agents that communicate via REST.

## Components

- `agent_service.py` – a simple FastAPI service exposing a `/chat` endpoint that returns a response from OpenAI based on the incoming message.
- `coordinator.py` – a FastAPI service that forwards a message to one or more agents and aggregates their replies.

Both services read `OPENAI_API_KEY` from the environment. The coordinator also uses an optional `AGENT_URLS` environment variable with comma-separated agent URLs.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run an agent service on port 8001:

```bash
uvicorn agent_service:app --port 8001
```

Run the coordinator and point it to the agent:

```bash
AGENT_URLS=http://localhost:8001 uvicorn coordinator:app --port 8000
```

Send a message:

```bash
curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{"message": "Hola"}'
```

The coordinator will return the response from the configured agent(s).
