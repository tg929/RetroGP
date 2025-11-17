# demo_utils.py
from gp_retro_repr import (
    Inventory, ReactionTemplateRegistry, ReactionTemplate,
)

from gp_retro_obj import ObjectiveSpec

def build_world_t1():
    stock = Inventory(["CC=O", "O"])
    reg = ReactionTemplateRegistry()
    reg.add(ReactionTemplate("T1", "[C:1]-[O:2]>>[C:1]=O.[O:2]", metadata={"family":"oxidation"}))
    target = "CCO"
    return stock, reg, target

def build_objectives_default():
    return {
        "solved":            ObjectiveSpec("solved", "max", weight=100.0),
        "route_len":         ObjectiveSpec("route_len", "min", weight=1.0),
        "valid_prefix":      ObjectiveSpec("valid_prefix", "max", weight=1.0),
        "sc_partial_reward": ObjectiveSpec("sc_partial_reward", "max", weight=5.0),
        "purch_frac":        ObjectiveSpec("purch_frac", "max", weight=2.0),
        "qed":               ObjectiveSpec("qed", "max", weight=1.0),
    }
