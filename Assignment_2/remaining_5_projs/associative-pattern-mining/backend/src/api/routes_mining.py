"""FastAPI routes for association rule mining, rule queries, and graph/scatter payloads."""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from ..core.state import app_state
from ..algorithms.apriori import AprioriMiner
from ..algorithms.fpgrowth import FPGrowthMiner
from ..algorithms.eclat import EclatMiner
from ..core.rules import RuleGenerator
from ..core.recommendations import CartRecommender
from .schemas import MiningRequest, MiningResponse, NetworkGraphResponse, GraphNode, GraphEdge, AssociationRuleDTO

router = APIRouter(prefix="/mine", tags=["Mining"])


@router.post("", response_model=MiningResponse)
def execute_mining(req: MiningRequest):
    """Executes frequent pattern mining using Apriori, FP-Growth, or ECLAT."""
    alg_lower = req.algorithm.lower().strip()
    if alg_lower == "apriori":
        miner = AprioriMiner()
    elif alg_lower in ("fpgrowth", "fp-growth", "fp_growth"):
        miner = FPGrowthMiner()
    elif alg_lower == "eclat":
        miner = EclatMiner()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported algorithm '{req.algorithm}'. Supported: 'apriori', 'fpgrowth', 'eclat'."
        )

    # Mine frequent itemsets
    mining_result = miner.mine(
        transactions=app_state.dataset.transactions,
        min_support=req.min_support,
        max_len=req.max_length
    )

    # Generate association rules
    rules = RuleGenerator.generate_rules(
        frequent_itemsets=mining_result.itemsets,
        min_confidence=req.min_confidence,
        min_lift=req.min_lift,
        max_consequent_len=1,
        prune_redundant=req.prune_redundant
    )

    # Update app state
    app_state.cached_mining_result = mining_result
    app_state.cached_rules = rules
    app_state.recommender = CartRecommender(rules=rules, dataset=app_state.dataset)

    # Prepare response
    top_rules = rules[:req.limit]
    rule_dtos = [AssociationRuleDTO(**r.to_dict()) for r in top_rules]

    return MiningResponse(
        algorithm=miner.name,
        execution_time_ms=round(mining_result.execution_time_ms, 2),
        num_transactions=mining_result.num_transactions,
        min_support=req.min_support,
        min_confidence=req.min_confidence,
        min_lift=req.min_lift,
        total_itemsets_found=mining_result.total_itemsets,
        total_rules_found=len(rules),
        itemsets_by_length=mining_result.itemsets_by_length(),
        rules=rule_dtos
    )


@router.get("/rules")
def query_rules(
    search: Optional[str] = Query(None, description="Search term in antecedent or consequent"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    min_lift: float = Query(0.0, ge=0.0),
    sort_by: str = Query("lift", description="Metric to sort by: lift, confidence, support, kulczynski, conviction, zhang"),
    limit: int = Query(50, ge=1, le=500)
) -> Dict[str, Any]:
    """Retrieves active mined association rules with customizable filtering and sorting."""
    rules = app_state.cached_rules
    filtered = []

    for r in rules:
        if r.metrics.confidence < min_confidence or r.metrics.lift < min_lift:
            continue
        if search:
            search_l = search.lower()
            in_ante = any(search_l in item.lower() for item in r.antecedent)
            in_cons = any(search_l in item.lower() for item in r.consequent)
            if not (in_ante or in_cons):
                continue
        filtered.append(r)

    # Sorting
    key_map = {
        "lift": lambda x: x.metrics.lift,
        "confidence": lambda x: x.metrics.confidence,
        "support": lambda x: x.metrics.support,
        "kulczynski": lambda x: x.metrics.kulczynski,
        "conviction": lambda x: x.metrics.conviction,
        "zhang": lambda x: x.metrics.zhangs_metric,
        "leverage": lambda x: x.metrics.leverage,
    }
    sort_fn = key_map.get(sort_by.lower(), lambda x: x.metrics.lift)
    filtered.sort(key=sort_fn, reverse=True)

    result_slice = filtered[:limit]
    return {
        "total_active_rules": len(rules),
        "matching_rules_count": len(filtered),
        "limit": limit,
        "sort_by": sort_by,
        "rules": [r.to_dict() for r in result_slice]
    }


@router.get("/graph", response_model=NetworkGraphResponse)
def get_rule_network_graph(
    min_lift: float = Query(1.10, ge=0.0),
    max_rules: int = Query(40, ge=5, le=150)
):
    """Generates node-edge graph structure for interactive Vis.js Association Rule Network."""
    rules = [r for r in app_state.cached_rules if r.metrics.lift >= min_lift][:max_rules]

    nodes_dict: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []

    n_tx = max(1, app_state.dataset.num_transactions)

    for rule in rules:
        # Collect all items involved
        for item in list(rule.antecedent) + list(rule.consequent):
            if item not in nodes_dict:
                supp = app_state.dataset.item_counts.get(item, 1) / n_tx
                # Value scales node radius
                scaled_val = max(12.0, min(35.0, supp * 100.0))
                nodes_dict[item] = GraphNode(
                    id=item,
                    label=item if len(item) <= 22 else item[:20] + "...",
                    value=scaled_val,
                    title=f"Item: {item}<br>Support: {round(supp * 100, 1)}%",
                    group="product"
                )

        # Build directed edge from each antecedent item to consequent
        # If antecedent has multiple items, create directed edges from all antecedents to consequent
        for ante_item in rule.antecedent:
            for cons_item in rule.consequent:
                edge_val = max(1.5, min(8.0, rule.metrics.lift))
                title_html = (
                    f"<b>{rule.rule_string}</b><br>"
                    f"Confidence: {round(rule.metrics.confidence * 100, 1)}%<br>"
                    f"Lift: {round(rule.metrics.lift, 2)}x<br>"
                    f"Kulczynski: {round(rule.metrics.kulczynski, 3)}<br>"
                    f"Support: {round(rule.metrics.support * 100, 2)}%"
                )
                edges.append(GraphEdge(
                    from_node=ante_item,
                    to_node=cons_item,
                    value=edge_val,
                    title=title_html,
                    label=f"{round(rule.metrics.confidence * 100)}%",
                    arrows="to"
                ))

    avg_lift = sum(r.metrics.lift for r in rules) / len(rules) if rules else 0.0
    avg_conf = sum(r.metrics.confidence for r in rules) / len(rules) if rules else 0.0

    return NetworkGraphResponse(
        nodes=list(nodes_dict.values()),
        edges=edges,
        rule_count=len(rules),
        node_count=len(nodes_dict),
        avg_lift=round(avg_lift, 2),
        avg_confidence=round(avg_conf, 2)
    )


@router.get("/matrix")
def get_rules_matrix(limit: int = 150) -> Dict[str, Any]:
    """Returns rules formatted for the 2D/3D Scatter Matrix Plot."""
    rules = app_state.cached_rules[:limit]
    return {
        "count": len(rules),
        "data": [
            {
                "rule": r.rule_string,
                "antecedent": r.antecedent_list,
                "consequent": r.consequent_list,
                "support": round(r.metrics.support, 4),
                "confidence": round(r.metrics.confidence, 4),
                "lift": round(r.metrics.lift, 3),
                "conviction": round(min(15.0, r.metrics.conviction), 3),
                "leverage": round(r.metrics.leverage, 4),
                "zhang": round(r.metrics.zhangs_metric, 3),
                "kulczynski": round(r.metrics.kulczynski, 3),
                "imbalance_ratio": round(r.metrics.imbalance_ratio, 3)
            }
            for r in rules
        ]
    }
