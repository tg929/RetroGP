"""Feasible execution wrapper."""
from typing import Any, Dict

from gp_retro_feas import FeasibleExecutor
from gp_retro_obj import RouteFitnessEvaluator
from gp_retro_repr import Program, Stop


def make_executor(reg, inventory, policy=None) -> FeasibleExecutor:
    return FeasibleExecutor(reg, inventory=inventory, policy=policy)


def evaluate_program(
    prog: Program,
    exe: FeasibleExecutor,
    evaluator: RouteFitnessEvaluator,
    target: str,
) -> Dict[str, Any]:
    try:
        route = exe.execute(prog, target_smiles=target)
    except Exception:
        safe_prog = Program([Stop()])
        route = exe.execute(safe_prog, target_smiles=target)
    fit = evaluator.evaluate(route)
    return {"program": prog, "route": route, "fitness": fit}
