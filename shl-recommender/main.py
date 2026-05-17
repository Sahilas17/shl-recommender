"""
SHL Assessment Recommender — FastAPI Service

Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → {"reply": "...", "recommendations": [...], "end_of_conversation": false}
"""

import logging
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator
from typing import Optional
from agent import chat

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent that recommends SHL Individual Test Assessments.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("content cannot be empty")
        if len(v) > 4000:
            raise ValueError("content too long (max 4000 chars)")
        return v.strip()


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        if len(v) > 20:
            raise ValueError("Too many messages (max 20)")
        if v[0].role != "user":
            raise ValueError("First message must be from 'user'")

        for i in range(1, len(v)):
            if v[i].role == v[i - 1].role:
                raise ValueError(
                    f"Messages must alternate roles. Got two '{v[i].role}' in a row at index {i}"
                )

        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ---------------------------------------------------------------------------
# Middleware — request timing
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({duration:.2f}s)"
    )

    return response


# ---------------------------------------------------------------------------
# Homepage
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SHL Assessment Recommender</title>

        <style>
            body {
                font-family: Arial;
                background: #0f172a;
                color: white;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }

            .container {
                width: 700px;
                background: #1e293b;
                padding: 30px;
                border-radius: 20px;
                box-shadow: 0 0 20px rgba(0,0,0,0.4);
            }

            h1 {
                text-align: center;
            }

            #chatbox {
                height: 400px;
                overflow-y: auto;
                background: #334155;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
            }

            input {
                width: 80%;
                padding: 12px;
                border-radius: 10px;
                border: none;
                outline: none;
                font-size: 16px;
            }

            button {
                width: 18%;
                padding: 12px;
                border: none;
                border-radius: 10px;
                background: #2563eb;
                color: white;
                font-size: 16px;
                cursor: pointer;
            }

            .user {
                color: #93c5fd;
                margin-bottom: 10px;
            }

            .bot {
                color: #86efac;
                margin-bottom: 20px;
            }
        </style>
    </head>

    <body>

        <div class="container">

            <h1>SHL Assessment Recommender</h1>

            <div id="chatbox"></div>

            <input
                type="text"
                id="message"
                placeholder="Describe the role you are hiring for..."
            />

            <button onclick="sendMessage()">
                Send
            </button>

        </div>

        <script>

            let messages = [];

            function escapeHTML(str) {
                return str.replace(/[&<>'"]/g, 
                    tag => ({
                        '&': '&amp;', '<': '&lt;', '>': '&gt;',
                        "'": '&#39;', '"': '&quot;'
                    }[tag])
                );
            }

            async function sendMessage() {

                const input = document.getElementById("message");

                const text = input.value;

                if (!text) return;

                const chatbox = document.getElementById("chatbox");

                chatbox.innerHTML += `
                    <div class="user">
                        <b>You:</b> ${escapeHTML(text)}
                    </div>
                `;

                messages.push({
                    role: "user",
                    content: text
                });

                input.value = "";

                const response = await fetch("/chat", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        messages: messages
                    })
                });

                const data = await response.json();

                chatbox.innerHTML += `
                    <div class="bot">
                        <b>AI:</b> ${escapeHTML(data.reply)}
                    </div>
                `;

                messages.push({
                    role: "assistant",
                    content: data.reply
                });

                chatbox.scrollTop = chatbox.scrollHeight;
            }

        </script>

    </body>
    </html>
    """

# ---------------------------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Readiness probe — returns 200 immediately."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Chat Endpoint
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Stateless chat endpoint.
    Caller passes the full conversation history.
    """

    messages = [
        {"role": m.role, "content": m.content}
        for m in req.messages
    ]

    try:
        result = chat(messages)

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)

        raise HTTPException(
            status_code=500,
            detail="Agent encountered an error. Please retry."
        )

    return ChatResponse(
        reply=result["reply"],

        recommendations=[
            Recommendation(
                name=r["name"],
                url=r["url"],
                test_type=r["test_type"]
            )
            for r in result.get("recommendations", [])
        ],

        end_of_conversation=result.get(
            "end_of_conversation",
            False
        )
    )
