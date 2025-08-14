# src/app.py  (append to your working version)
from fastapi import FastAPI
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from .planner import build_plan  # NEW
from .nlp_parser import parse_text

app = FastAPI(title="UAE Expat Financial Planner")

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
