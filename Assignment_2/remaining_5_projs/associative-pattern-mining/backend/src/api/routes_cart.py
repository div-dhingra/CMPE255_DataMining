"""FastAPI routes for interactive market basket cart playground and recommendations."""

from fastapi import APIRouter
from typing import Dict, Any, List
from ..core.state import app_state
from .schemas import CartRecommendationRequest, CartRecommendationResponse

router = APIRouter(prefix="/cart", tags=["Cart Recommendations"])


@router.get("/catalog")
def get_catalog_items(limit: int = 40) -> Dict[str, Any]:
    """Returns available catalog items and their frequency for the cart playground selector."""
    items = list(app_state.dataset.item_counts.items())[:limit]
    n_tx = max(1, app_state.dataset.num_transactions)

    return {
        "total_items": len(items),
        "items": [
            {
                "name": name,
                "count": count,
                "support": round(count / n_tx, 4),
                "frequency_pct": round((count / n_tx) * 100, 1)
            }
            for name, count in items
        ]
    }


@router.post("/recommend", response_model=CartRecommendationResponse)
def get_cart_recommendations(req: CartRecommendationRequest):
    """Generates association-driven item recommendations based on the active shopping cart contents."""
    if not app_state.recommender:
        from ..core.recommendations import CartRecommender
        app_state.recommender = CartRecommender(rules=app_state.cached_rules, dataset=app_state.dataset)

    result = app_state.recommender.recommend(
        cart_items=req.cart_items,
        top_k=req.top_k,
        min_lift=req.min_lift
    )
    return CartRecommendationResponse(**result)
