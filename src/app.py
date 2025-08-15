# src/app.py  (append to your working version)
from fastapi import FastAPI
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from .planner import build_plan  # NEW
from .nlp_parser import parse_text
from src.nlp_ml_infer import parse_goals

app = FastAPI(title="UAE Expat Financial Planner")

# Uncomment to allow browser apps on localhost, etc.
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# ---------- Schemas ----------
class ParseRequest(BaseModel):
    text: str = Field(..., description="User's free-form goal text")


class Goal(BaseModel):
    original_text: str
    goal_type: str
    classifier_confidence: float
    amount: Optional[float] = None
    currency: Optional[str] = None
    timeframe_years: Optional[int] = None
    timeframe_months: Optional[int] = None
    timeframe_months_total: Optional[int] = None
    location: Optional[str] = None
    notes: Dict[str, Optional[str]]


class ParseResponse(BaseModel):
    schema_version: str
    input_text: str
    goals: List[Goal]


class PlanFromText(BaseModel):
    profile: Dict[str, Any]
    goals_text: str
    frs_score: Optional[float] = None
    mode_per_goal: Optional[str] = "assumed_return"   # "assumed_return" | "required_return"
    monthly_budget_map: Optional[Dict[int, float]] = None
    lstm_symbols: Optional[List[str]] = None

@app.post("/plan/from-text-advanced")
def plan_from_text_advanced(payload: PlanFromText):
    parsed = parse_text(payload.goals_text)
    goals = parsed.get("goals", [])
    # Inject your real LSTM predictor here if available:
    result = build_plan(
        profile=payload.profile,
        goals=goals,
        frs_score=payload.frs_score,
        mode_per_goal=payload.mode_per_goal,
        monthly_budget_map=payload.monthly_budget_map,
        lstm_symbols=payload.lstm_symbols,
        # lstm_predictor=your_lstm_predictor_fn
    )
    return {"parsed_goals": parsed, "plan": result}

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/nlp/parse", response_model=ParseResponse)
def nlp_parse(req: ParseRequest):
    """
    Call your ML pipeline:
    - Sentence split
    - (Masked) classification for goal_type
    - spaCy NER for amount/currency/time/location
    - Post-process & structure
    """
    result: Dict[str, Any] = parse_goals(req.text)
    # Pydantic will validate/shape the response for us
    return result
