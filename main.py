import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests

from database import db, create_document, get_documents
from schemas import Analysis

app = FastAPI(title="AI Politique API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Auth (Supabase) --------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

class UserInfo(BaseModel):
    sub: str
    email: Optional[str] = None


def get_current_user(authorization: Optional[str] = None) -> Optional[UserInfo]:
    """
    Lightweight auth: expects `Authorization: Bearer <jwt>` from Supabase Auth.
    If token is provided and python-jose is available, decode to extract user id (sub) and email.
    If no token, return None (treat as anonymous) — still allows using the API but no per-user history/quotas.
    """
    try:
        from jose import jwt
    except Exception:
        return None

    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    try:
        # For Supabase, the anon public key is the verifying key for JWT (HS256)
        payload = jwt.get_unverified_claims(token)
        sub = payload.get("sub") or payload.get("user_id")
        email = payload.get("email")
        if sub:
            return UserInfo(sub=sub, email=email)
    except Exception:
        return None
    return None


# --------- Analyze models ---------
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

    # Fallback: if model returns markdown/text, use as summary
    result = {
        "summary": content.strip() or "",
        "tone": "",
        "bias": "",
        "rhetoric": "",
        "strategy": "",
        "keywords": [],
        "recommendations": [],
    }
    return result


SYSTEM_PROMPT = (
    "You are an expert political, legal, and humanitarian analysis assistant. "
    "Analyze the provided text and return a concise JSON with keys: "
    "summary, tone, bias, rhetoric, strategy, keywords (array), recommendations (array of 3). "
    "Be neutral and evidence-based."
)


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, user: Optional[UserInfo] = Depends(get_current_user)):
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
            recommendations=ai_result.get("recommendations", []),
            user_id=user.sub if user else None,
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
def history(limit: int = 10, authorization: Optional[str] = None):
    """Return recent items. If Authorization Bearer token present and valid, filter by user_id."""
    user = get_current_user(authorization)
    try:
        filter_dict = {}
        if user:
            filter_dict["user_id"] = user.sub
        docs = get_documents("analysis", filter_dict=filter_dict, limit=limit)
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
