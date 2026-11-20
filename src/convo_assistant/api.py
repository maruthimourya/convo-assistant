import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from .llm import LLM, make_llm

app = FastAPI(title="Conversation Assistant")
SESSIONS: dict[str, list[dict]] = {}
SYSTEM = {
    "role": "system",
    "content": "You are a helpful financial education assistant. Do not give investment advice.",
}


class MessageIn(BaseModel):
    text: str


class MessageOut(BaseModel):
    reply: str
    turns: int


def get_llm() -> LLM:
    return make_llm()


@app.post("/sessions")
def create_session() -> dict:
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = [SYSTEM]
    return {"session_id": session_id}


@app.post("/sessions/{session_id}/messages", response_model=MessageOut)
def send_message(
    session_id: str, msg: MessageIn, llm: Annotated[LLM, Depends(get_llm)]
):
    history = SESSIONS.get(session_id)
    if history is None:
        raise HTTPException(404, "session not found")
    history.append({"role": "user", "content": msg.text})
    answer = llm.reply(history)
    history.append({"role": "assistant", "content": answer})
    return MessageOut(reply=answer, turns=(len(history) - 1) // 2)


@app.get("/sessions/{session_id}")
def get_history(session_id: str) -> list[dict]:
    if session_id not in SESSIONS:
        raise HTTPException(404, "session not found")
    return SESSIONS[session_id][1:]
