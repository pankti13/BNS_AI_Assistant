from typing import Union
from fastapi import FastAPI

app = FastAPI()

# fetch user-input query => check whether it is a "scenario" or not => return top 5 scenarios (using nearest neighbours using embeddings) in json form else let gemini handle the greeting msgs
@app.get("/")
def read_root():
    return {"Hello World"}

