import os
from typing import List

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

AGENT_URLS = [url.strip() for url in os.getenv("AGENT_URLS", "").split(',') if url.strip()]

app = FastAPI(title="Coordinator")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    responses: List[str]

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Forward the message to each configured agent and return their replies."""
    if not AGENT_URLS:
        return ChatResponse(responses=["No agents configured."])

    responses = []
    async with httpx.AsyncClient() as client:
        for url in AGENT_URLS:
            try:
                r = await client.post(f"{url}/chat", json={"message": request.message}, timeout=10)
                r.raise_for_status()
                data = r.json()
                responses.append(data.get("response", ""))
            except Exception as exc:
                responses.append(f"Error contacting {url}: {exc}")
    return ChatResponse(responses=responses)
