"""中医处方审核与中医药大模型安全评测基准"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.eval_api import router as eval_router

app = FastAPI(
    title="TCM-LLM-SafetyEval",
    version="0.3.0",
    description="中医处方审核与中医药大模型安全评测基准",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register evaluation API router
app.include_router(eval_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "TCM-LLM-SafetyEval", "version": "0.2.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8029)
