"""API v1 聚合路由."""

from fastapi import APIRouter
from .auth import router as auth_router
from .literature import router as literature_router
from .knowledge_graph import router as kg_router
from .property import router as property_router
from .microscopy import router as microscopy_router
from .crystal import router as crystal_router
from .learning import router as learning_router

router = APIRouter(prefix="/api/v1")
router.include_router(auth_router)
router.include_router(literature_router)
router.include_router(kg_router)
router.include_router(property_router)
router.include_router(microscopy_router)
router.include_router(crystal_router)
router.include_router(learning_router)
