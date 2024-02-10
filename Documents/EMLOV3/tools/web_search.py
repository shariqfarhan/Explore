from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

class SearchQuery(BaseModel):
    query: str

def duckduckgo_search(query: str):
    """Perform a search on DuckDuckGo and return the results."""
    api_url = "https://api.duckduckgo.com/"
    params = {
        'q': query,
        'format': 'json',
        'pretty': '1'
    }
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        results = response.json()
        return results
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/search/")
async def search(search_query: SearchQuery):
    """Endpoint to perform a search using DuckDuckGo."""
    results = duckduckgo_search(search_query.query)
    return results
