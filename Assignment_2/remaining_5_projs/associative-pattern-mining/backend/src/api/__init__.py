"""API package initialization."""

from .routes_crisp_dm import router as crisp_dm_router
from .routes_data import router as data_router
from .routes_mining import router as mining_router
from .routes_autoresearch import router as autoresearch_router
from .routes_cart import router as cart_router

__all__ = [
    "crisp_dm_router",
    "data_router",
    "mining_router",
    "autoresearch_router",
    "cart_router"
]
