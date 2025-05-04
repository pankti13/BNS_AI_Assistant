from fastapi import APIRouter, HTTPException
from models import QueryInput
from services.gemini_service import generate_gemini_response
from services.scenario_service import ScenarioService

router = APIRouter()
scenario_service = ScenarioService()

@router.post("/query")
async def handle_query(payload: QueryInput):
    query = payload.query.strip()
    history = payload.history
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        if scenario_service.is_scenario_query(query, history):
            return {
                "type": "scenario", 
                "results": scenario_service.get_top_scenarios(query)
            }
        else:
            return {
                "type": "general", 
                "results": generate_gemini_response(query, history)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
