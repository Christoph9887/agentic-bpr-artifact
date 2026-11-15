"""MABPRState — the blackboard (METHOD §9.3)."""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

class MABPRState(TypedDict, total=False):
    """One redesign round on the blackboard. ``total=False`` so every node may"""

    round: int
    paths: Dict[str, str]
    sim_config: Dict[str, Any]

    mode: str
    current_plan: List[str]
    last_agg: dict
    tried_plans: Annotated[List[dict], operator.add]
    anti_combos: List[List[str]]

    build_history: Dict[str, List[dict]]

    deleter_probes: Dict[str, dict]

    clean_probe: Dict[str, object]

    measured_designs: Dict[str, dict]

    harmful_history: Dict[str, List[dict]]

    candidate_ledger: Annotated[List[dict], operator.add]

    activity_ledger: Annotated[List[dict], operator.add]
    design_key: str
    apply_dir: str

    directive_followthrough: Dict[str, object]

    proposal: dict

    explanation: dict
    knowledge_log: dict

    gate_ruling: dict
    blocked_candidates: List[str]

    compliance_advisory: Annotated[List[dict], operator.add]

    compliance_final: dict

    gate_enforced: bool
    calibration: dict
    contradictions: Annotated[List[dict], operator.add]

    max_rounds: int

    best_agg: dict
    best_median_h: float
    best_median_std: float
    rounds_without_improvement: int
    improvement: dict

    improvement_z: float

    base_agg: dict

    redes_agg: dict

    footprints: Dict[str, dict]
    bundle: dict

    hypotheses: List[dict]
    read_log: dict
    confirmation_request: Optional[dict]
    confirmation_granted: bool
    confirmation_reason: str
    confirmation_plan: List[str]
    confirmation_bundle: Optional[dict]

    assessment: dict

    sim_budget: int
    confirm_budget: int
    confirm_per_round: int

    trace: Annotated[List[dict], operator.add]
