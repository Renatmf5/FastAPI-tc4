from fastapi import APIRouter
from .endpoints import manage_metrics, manage_calls

api_router = APIRouter()

api_router.include_router(manage_metrics.router, prefix="/callBacktests", tags=["getBacktests"])
api_router.include_router(manage_calls.router, prefix="/callPredicts", tags=["getPredicts"])

@api_router.get("/")
async def root():
    return {"message": "Hello World"}