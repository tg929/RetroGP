# run_obj_demo.py
# 验证 gp_retro_obj 模块 + 和 gp_retro_repr / gp_retro_feas 的整合使用

import sys
import traceback

print(">>> run_obj_demo.py: starting ...")

from gp_retro_repr import (
    Program, Select, ApplyTemplate, Stop,
)
from gp_retro_feas import FeasibleExecutor
from gp_retro_obj import (
    RouteFitnessEvaluator,
    epsilon_lexicase_select,
    nsga2_survivor_selection,
)
from demo_utils import build_world_t1, build_objectives_default

# 新增：从补丁包里引入 SCScore 封装
from llm_syn_planner.scscore_reward import make_scscore


# ---------------------------------------------------------------------------
# 1. 定义一个简单的“审计函数”给 RouteFitnessEvaluator 用
# ---------------------------------------------------------------------------
def make_audit_fn(stock, target_smiles: str):
    """
    RouteFitnessEvaluator 期望有一个 audit_fn(route)，返回：
        is_solved, first_invalid_molecule_set, current_molecule_set,
        n_steps, n_valid_steps
    """
    def audit_route(route):
        if route.steps:
            final_set = route.steps[-1].updated_molecule_set
            n_steps = len(route.steps)
        else:
            final_set = [target_smiles]
            n_steps = 0
        is_solved = all(stock.is_purchasable(m) for m in final_set)
        return {
            "is_solved": is_solved,
            "first_invalid_molecule_set": [] if is_solved else list(final_set),
            "current_molecule_set": list(final_set),
            "n_steps": n_steps,
            "n_valid_steps": n_steps,
        }
    return audit_route


# ---------------------------------------------------------------------------
# 2. 主流程：执行两条 DP 程序 → 计算多目标适应度 → 演示选择算子
# ---------------------------------------------------------------------------
def main():
    print(">>> main() entered in run_obj_demo.py")

    # 2.1 世界与目标（统一使用 demo_utils）
    stock, reg, target = build_world_t1()
    exe = FeasibleExecutor(reg, inventory=stock)
    print(f"Target SMILES: {target}")
    print(f"Stock molecules: {list(stock)}")

    # === 新增：初始化 SCScore 模型（一次即可） ===
    # 如果你的 scscore 模型需要指定模型目录，可以在这里传 model_dir 或设置 SCSCORE_MODEL_DIR 环境变量
    sc_fn, _ = make_scscore()  # 我们只用到第一个返回值：scscore_fn(smiles) -> float

    # 2.2 定义两条 DP 程序
    prog_good = Program(
        [Select(0), ApplyTemplate("T1", rational="good_demo"), Stop()]
    )
    prog_bad = Program([Stop()])

    programs = [("good_route", prog_good), ("bad_route", prog_bad)]

    # 2.3 多目标配置 & 适应度评估器（统一使用 demo_utils）
    specs = build_objectives_default()
    audit_fn = make_audit_fn(stock, target_smiles=target)

    evaluator = RouteFitnessEvaluator(
        objective_specs=specs,
        purchasable_fn=stock.is_purchasable,
        audit_fn=audit_fn,
        scscore_fn=sc_fn,      # ★ 关键：把真实 SCScore 函数接进来
        target_smiles=target,
    )

    population = []

    # 2.4 执行每个程序，打印路线 JSON 和多目标适应度
    for name, prog in programs:
        print("\n" + "=" * 80)
        print(f"[{name}] 执行 DP 程序")

        route = exe.execute(prog, target_smiles=target)
        print("Route JSON:")
        print(route.to_json())
        print("Route is_solved(stock):", route.is_solved(stock))

        fit = evaluator.evaluate(route)
        print("\nObjectives:")
        for k, v in fit.objectives.items():
            print(f"  {k:18s}: {v:.4f}")
        print("Scalar fitness:", fit.scalar)

        population.append(
            {"name": name, "program": prog, "route": route, "fitness": fit}
        )

    # 2.5 多目标选择算子：ε-lexicase（父代）+ NSGA-II（生存者）
    senses = {k: spec.direction() for k, spec in specs.items()}
    objective_keys = list(specs.keys())

    print("\n" + "=" * 80)
    print("[epsilon-lexicase] 选择父代")
    parents = epsilon_lexicase_select(
        population,
        senses=senses,
        objective_keys=objective_keys,
        eps_quantile=0.10,
        n_parents=2,
    )
    print("被选中的父代个体：", [ind["name"] for ind in parents])

    print("\n[NSGA-II] 生存者选择 (k=1)")
    survivors = nsga2_survivor_selection(
        population,
        k=1,
        senses=senses,
        objective_keys=objective_keys,
    )
    print("保留下来的个体：", [ind["name"] for ind in survivors])

    print("\n>>> run_obj_demo.py finished normally.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
