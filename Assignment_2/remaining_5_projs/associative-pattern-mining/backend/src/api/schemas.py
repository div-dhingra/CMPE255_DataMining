"""Pydantic V2 schemas for typed API request and response bodies."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DatasetSummaryResponse(BaseModel):
    num_transactions: int
    num_unique_items: int
    total_items_purchased: int
    avg_basket_size: float
    median_basket_size: float
    std_basket_size: Optional[float] = 0.0
    min_basket_size: int
    max_basket_size: int
    density_percent: float
    sparsity_percent: float
    top_items: List[Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SyntheticGenerateRequest(BaseModel):
    num_transactions: int = Field(default=800, ge=50, le=5000)
    avg_basket_size: float = Field(default=4.0, ge=1.5, le=10.0)
    noise_level: float = Field(default=0.20, ge=0.0, le=1.0)
    seed: Optional[int] = 42


class MiningRequest(BaseModel):
    algorithm: str = Field(default="fpgrowth", description="apriori, fpgrowth, or eclat")
    min_support: float = Field(default=0.03, ge=0.001, le=1.0)
    min_confidence: float = Field(default=0.40, ge=0.0, le=1.0)
    min_lift: float = Field(default=1.10, ge=0.0)
    max_length: int = Field(default=3, ge=2, le=6)
    prune_redundant: bool = True
    limit: int = Field(default=50, ge=1, le=500)


class AssociationRuleDTO(BaseModel):
    rule_string: str
    antecedent: List[str]
    consequent: List[str]
    support: float
    antecedent_support: float
    consequent_support: float
    confidence: float
    lift: float
    conviction: float
    leverage: float
    zhangs_metric: float
    kulczynski: float
    imbalance_ratio: float


class MiningResponse(BaseModel):
    algorithm: str
    execution_time_ms: float
    num_transactions: int
    min_support: float
    min_confidence: float
    min_lift: float
    total_itemsets_found: int
    total_rules_found: int
    itemsets_by_length: Dict[int, int]
    rules: List[AssociationRuleDTO]


class GraphNode(BaseModel):
    id: str
    label: str
    value: float  # scaled by support
    title: str
    group: Optional[str] = None


class GraphEdge(BaseModel):
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    value: float  # scaled by lift
    title: str
    label: Optional[str] = None
    arrows: str = "to"

    class Config:
        populate_by_name = True


class NetworkGraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    rule_count: int
    node_count: int
    avg_lift: float
    avg_confidence: float


class AutoresearchTriggerRequest(BaseModel):
    max_steps: int = Field(default=10, ge=1, le=100)
    cooling_rate: float = Field(default=0.94, ge=0.80, le=0.99)
    max_stagnation: int = Field(default=5, ge=2, le=15)


class AutoresearchStepResponse(BaseModel):
    iteration: int
    action: str
    candidate_fitness: float
    current_fitness: float
    best_fitness: float
    fitness_delta: float
    accepted: bool
    temperature: float
    execution_time_ms: float
    configuration: Dict[str, Any]
    details: str
    rule_count: int
    mean_lift: float
    mean_kulczynski: float


class CartRecommendationRequest(BaseModel):
    cart_items: List[str] = Field(default_factory=list)
    top_k: int = Field(default=6, ge=1, le=20)
    min_lift: float = Field(default=1.05, ge=0.5)


class CartRecommendationResponse(BaseModel):
    cart_items: List[str]
    num_cart_items: int
    recommendations_count: int
    recommendations: List[Dict[str, Any]]
    projected_cross_sell_uplift_pct: float
    matched_rules_count: int
    mode: Optional[str] = None
