from pydantic import BaseModel
from typing import List, Dict, Any

class QueryInput(BaseModel):
    query: str
    history: List[Dict[str, Any]] = []
