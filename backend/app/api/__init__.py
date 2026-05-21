from fastapi import APIRouter

from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.operations import router as operations_router
from backend.app.api.routes.research import router as research_router
from backend.app.api.routes.workspace import router as workspace_router


api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(operations_router)
api_router.include_router(research_router)
api_router.include_router(workspace_router)
