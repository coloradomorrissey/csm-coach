from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import requests, time, os

app = FastAPI(title="CSM Coach (Local)")

# Paths to prompts (robust to where you run uvicorn)
BASE = Path(__file__).resolve().parent.parent
RUBRIC = (BASE / "prompts" / "lesson_rubric.md").read_text(encoding="utf-8")
COACH  = (BASE / "prompts" / "coach_style.md").read_text(encoding="utf-8")

# Ollama config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")  # try "phi4" if llama3.1 is slow

class LessonRequest(BaseModel):
    stage: str                # onboarding | adoption | value_expansion | risk | renewal
    experience_level: str     # new | mid | senior
    scenario: str             # learner's free-text situation

def ollama_generate(prompt: str, model: str = MODEL, max_retries: int = 2) -> str:
    """Call Ollama's simple generate API. Retries a couple of times for transient errors."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    last = None
    for i in range(max_retries + 1):
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        last = r
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        time.sleep(1 + i)
    raise RuntimeError(f"Ollama error: {last.status_code} {last.text[:300]}")

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/generate_lesson")
def generate_lesson(req: LessonRequest):
    """
    Compose a single prompt: rubric + learner context.
    Return lesson markdown.
    """
    prompt = f"""{RUBRIC}

Context for personalization:
- Stage: {req.stage}
- Experience: {req.experience_level}
- Scenario: {req.scenario}

Return MARKDOWN only. Use the exact sections from the rubric.
"""
    content = ollama_generate(prompt)
    return {"ok": True, "model": MODEL, "lesson_markdown": content}
