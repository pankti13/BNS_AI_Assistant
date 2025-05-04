from pydantic import BaseModel

class QueryInput(BaseModel):
    query: str
