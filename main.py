import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests

from database import db, create_document, get_documents
from schemas import Analysis

app = FastAPI(title="AI Politique API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    domain: Optional[str] = Field(None, description="political | legal | humanitarian")
    language: Optional[str] = Field(None, description="Input language, auto-detect if None")

class AnalyzeResponse(BaseModel):
    summary: str
    tone: str
    bias: str
    rhetoric: str
    strategy: str
    keywords: List[str]
    recommendations: List[str]

class AnalysisRecord(AnalyzeResponse):
    id: str


@app.get("/")
def read_root():
    return {"message": "AI Politique Backend is running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["connection_status"] = "Connected"
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


# --- AI Provider helpers ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def call_openai(system_prompt: str, user_text: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        # Fallback simple heuristic if no API key is provided
        return {
            "summary": user_text[:200] + ("..." if len(user_text) > 200 else ""),
            "tone": "neutral",
            "bias": "not enough context",
            "rhetoric": "informational",
            "strategy": "exposition",
            "keywords": list({w.strip('.,;:!?').lower() for w in user_text.split() if len(w) > 5})[:6],
            "recommendations": [
                "Collect more context and sources",
                "Compare with independent fact-checks",
                "Assess policy feasibility and stakeholders",
            ],
        }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.2,
    }
    url = f"{OPENAI_BASE_URL}/chat/completions"
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {r.text[:200]}")
    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Expect a structured block; try to parse simple fields heuristically
    # For robustness, we use simple markers; if not found, return content as summary
    result = {
        "summary": content,
        "tone": "", "bias": "", "rhetoric": "", "strategy": "",
        "keywords": [], "recommendations": []
    }
    return result


SYSTEM_PROMPT = (
    "You are an expert political, legal, and humanitarian analysis assistant. "
    "Analyze the provided text and return a concise JSON with keys: "
    "summary, tone, bias, rhetoric, strategy, keywords (array), recommendations (array of 3). "
    "Be neutral and evidence-based."
)


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not req.text or len(req.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Please provide at least 20 characters of text.")

    ai_result = call_openai(SYSTEM_PROMPT, req.text)

    # Persist to DB if available
    try:
        record = Analysis(
            input_text=req.text,
            domain=req.domain,
            language=req.language,
            summary=ai_result.get("summary", ""),
            tone=ai_result.get("tone", "neutral"),
            bias=ai_result.get("bias", "unknown"),
            rhetoric=ai_result.get("rhetoric", "informational"),
            strategy=ai_result.get("strategy", "exposition"),
            keywords=ai_result.get("keywords", []),
            recommendations=ai_result.get("recommendations", [])
        )
        try:
            create_document("analysis", record)
        except Exception:
            pass
    except Exception:
        pass

    return AnalyzeResponse(
        summary=ai_result.get("summary", ""),
        tone=ai_result.get("tone", "neutral"),
        bias=ai_result.get("bias", "unknown"),
        rhetoric=ai_result.get("rhetoric", "informational"),
        strategy=ai_result.get("strategy", "exposition"),
        keywords=ai_result.get("keywords", []),
        recommendations=ai_result.get("recommendations", []),
    )


class HistoryItem(BaseModel):
    id: str
    summary: str
    tone: str
    bias: str


@app.get("/api/history")
def history(limit: int = 10):
    try:
        docs = get_documents("analysis", limit=limit)
        # Convert ObjectId to string and map minimal fields
        items = []
        for d in docs:
            d_id = str(d.get("_id"))
            items.append({
                "id": d_id,
                "summary": d.get("summary", ""),
                "tone": d.get("tone", ""),
                "bias": d.get("bias", ""),
            })
        return {"items": items}
    except Exception:
        return {"items": []}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
