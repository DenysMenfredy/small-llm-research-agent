from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from research_agent.agent.orchestrator import ResearchAgent
import logging
import asyncio

logger = logging.getLogger(__name__)

app = FastAPI(title="Research Agent API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

agent = ResearchAgent()

@app.post("/research", response_model=QueryResponse)
async def research_endpoint(request: QueryRequest):
    try:
        answer = await asyncio.wait_for(agent.run(request.query), timeout=180.0)
        return QueryResponse(answer=answer)
    except asyncio.TimeoutError:
        logger.error("Research request timed out")
        raise HTTPException(status_code=504, detail="Request timed out after 3 minutes")
    except Exception as e:
        logger.error(f"Research endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)