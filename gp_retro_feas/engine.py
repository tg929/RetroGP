
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Iterable
from functools import lru_cache

from gp_retro_repr import ReactionTemplateRegistry, Inventory

@dataclass
class FeasibilityResult:
    ok: bool
    template_id: str
    product: str
    chosen_reactants: List[str] = field(default_factory=list)
    all_reactant_sets: List[List[str]] = field(default_factory=list)
    reason: str = ""
    availability_ok: Optional[bool] = None

class FeasibilityEngine:
    """
    Wraps rdchiral applicability and inventory gating.
    - apply(template, product) -> reactant sets
    - choose one set by a simple heuristic (fewest reactants), optionally require all purchasable
    """
    def __init__(self, reg: ReactionTemplateRegistry, inventory: Optional[Inventory] = None):
        self.reg = reg
        self.inventory = inventory

    @lru_cache(maxsize=200000)
    def _apply(self, template_id: str, product_smiles: str) -> List[List[str]]:
        tmpl = self.reg.get(template_id)
        return tmpl.apply_to_product(product_smiles)

    def check_and_choose(self,
                         template_id: str,
                         product_smiles: str,
                         require_all_purchasable: bool = False,
                         max_reactants: Optional[int] = None) -> FeasibilityResult:
        try:
            outs = self._apply(template_id, product_smiles)
        except Exception as e:
            return FeasibilityResult(
                ok=False, template_id=template_id, product=product_smiles,
                reason=f"apply_failed:{type(e).__name__}:{e}"
            )
        if not outs:
            return FeasibilityResult(
                ok=False, template_id=template_id, product=product_smiles,
                reason="no_reactant_set"
            )

        # simple heuristic: keep sets filtered by max_reactants, pick shortest length first
        candidates = [rs for rs in outs if (max_reactants is None or len(rs) <= max_reactants)]
        if not candidates:
            return FeasibilityResult(
                ok=False, template_id=template_id, product=product_smiles,
                all_reactant_sets=outs, reason="exceed_max_reactants"
            )
        candidates.sort(key=len)
        chosen = candidates[0]

        avail_ok = None
        if require_all_purchasable and self.inventory is not None:
            avail_ok = all(self.inventory.is_purchasable(s) for s in chosen)
            if not avail_ok:
                return FeasibilityResult(
                    ok=False, template_id=template_id, product=product_smiles,
                    all_reactant_sets=candidates, reason="inventory_violation", availability_ok=False
                )

        return FeasibilityResult(
            ok=True, template_id=template_id, product=product_smiles,
            chosen_reactants=chosen, all_reactant_sets=candidates,
            reason="ok", availability_ok=avail_ok
        )
