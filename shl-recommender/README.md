# SHL Assessment Recommender

A conversational AI agent that helps hiring managers find the right SHL Individual Test Assessments through natural dialogue.

## Architecture

```
User → POST /chat → FastAPI → Agent (Google Gemini 2.0 Flash) → Response + Recommendations
                                    ↑
                              TF-IDF Retrieval
                              (shl_catalog.json)
```

### Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Retrieval | TF-IDF (sklearn) | Zero external dependencies, no model download, fast startup, works offline, easily reproducible |
| LLM | Google Gemini 2.0 Flash | Fast, cost-effective, good instruction-following, works well for structured JSON output |
| State management | Stateless (caller sends full history) | Matches spec exactly; no server-side session storage |
| Response format | Strict JSON from LLM | Deterministic schema validation; evaluator-safe |
| URL validation | Allowlist check (`shl.com`) | Prevents hallucinated URLs from reaching the evaluator |

## Endpoints

### `GET /health`
Returns `{"status": "ok"}` with HTTP 200.

### `POST /chat`
**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Got it. Here are 5 assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

`recommendations` is an empty array when the agent is clarifying or refusing. `end_of_conversation` is `true` only when the task is complete.

### Test type codes
- `A` — Ability/Aptitude (reasoning, cognitive)
- `K` — Knowledge/Skills (technical, domain)
- `P` — Personality (questionnaires)
- `S` — Simulation (situational judgment, coding)

## Local setup

### Prerequisites
- Python 3.11+
- A Google Gemini API key (free tier available)

### Steps

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd shl-recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key (get free key from https://aistudio.google.com/apikey)
export GOOGLE_API_KEY=your_gemini_api_key_here

# 4. Build the retrieval index (run once)
python build_index.py

# 5. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000

# 6. Test
curl http://localhost:8000/health
python tests.py
```

## Deployment on Render (recommended, free tier)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Configure:
    - **Build Command:** `pip install -r requirements.txt && python build_index.py`
    - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
    - **Environment Variable:** `GOOGLE_API_KEY` = your free key from [Google AI Studio](https://aistudio.google.com/apikey)
5. Click Deploy
6. Your service URL will be `https://shl-recommender-xxxx.onrender.com`

> **Note:** Render free tier has cold starts (~30s). The evaluator allows 2 minutes for the first `/health` call.

## Deployment with Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GOOGLE_API_KEY=your_gemini_api_key shl-recommender
```

## Agent behaviors

| Behavior | Trigger | Response |
|---|---|---|
| **Clarify** | Vague query ("I need an assessment") | Asks 1-2 targeted questions; `recommendations: []` |
| **Recommend** | Enough context (role + level) | Returns 1-10 grounded assessments |
| **Refine** | User changes constraints | Updates shortlist in same conversation |
| **Compare** | "What is the difference between X and Y?" | Grounded comparison from catalog data |
| **Refuse scope** | Off-topic (salary, legal, general HR) | Polite refusal; `recommendations: []` |
| **Refuse injection** | Prompt injection attempts | Polite refusal without complying |

## Refreshing the catalog

The catalog is pre-committed as `shl_catalog.json`. To refresh:

```bash
python scraper.py       # re-scrapes SHL website → updates shl_catalog.json
python build_index.py   # rebuilds retrieval index → updates shl_retrieval.pkl
```

## Running tests

```bash
# Start server first
uvicorn main:app --port 8000 &

# Run all tests
python tests.py
```

Tests cover: health, vague query clarification, Java dev scenario, personality/leadership, multi-turn refinement, comparison questions, out-of-scope refusals, prompt injection, job description matching.
