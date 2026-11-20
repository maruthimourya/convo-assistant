from fastapi.testclient import TestClient

from convo_assistant.api import app, get_llm


class FakeLLM:
    def reply(self, history):
        return f"echo: {history[-1]['content']}"


app.dependency_overrides[get_llm] = lambda: FakeLLM()
client = TestClient(app)


def test_conversation_keeps_history():
    sid = client.post("/sessions").json()["session_id"]
    r1 = client.post(f"/sessions/{sid}/messages", json={"text": "hi"})
    r2 = client.post(f"/sessions/{sid}/messages", json={"text": "what is an ETF?"})
    assert r1.json()["reply"] == "echo: hi"
    assert r2.json()["turns"] == 2
    assert len(client.get(f"/sessions/{sid}").json()) == 4


def test_unknown_session_returns_404():
    assert (
        client.post("/sessions/nope/messages", json={"text": "hi"}).status_code == 404
    )
