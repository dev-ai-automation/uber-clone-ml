"""
Main API router for v1 endpoints.
"""

from fastapi import APIRouter

from app.api.v1 import users, rides, drivers, matching

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(rides.router, prefix="/rides", tags=["rides"])
api_router.include_router(drivers.router, prefix="/drivers", tags=["drivers"])
api_router.include_router(matching.router, prefix="/matching", tags=["matching"])
