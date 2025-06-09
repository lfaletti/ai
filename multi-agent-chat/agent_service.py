import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Simple Agent Service")

# store conversation in memory for up to 15 minutes
CONVERSATION: list = []
START_TIME: float | None = None
MEMORY_SECONDS = 15 * 60

def _get_history() -> list:
    """Return current conversation history, resetting after 15 minutes."""
    global CONVERSATION, START_TIME
    now = time.time()
    if START_TIME is None or now - START_TIME > MEMORY_SECONDS:
        START_TIME = now
        CONVERSATION = []
    return CONVERSATION

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Return a chat completion using the stored conversation history."""
    history = _get_history()
    history.append({"role": "user", "content": request.message})

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=history,
        max_tokens=128,
    )

    reply = completion.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": reply})
    return ChatResponse(response=reply)
