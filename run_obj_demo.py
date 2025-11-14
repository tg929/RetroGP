# run_obj_demo.py
# 验证 gp_retro_obj 模块 + 和 gp_retro_repr / gp_retro_feas 的整合使用

import sys
import traceback

# 只要脚本被执行，这句一定会打印出来，方便确认脚本本身是否在运行
print(">>> run_obj_demo.py: starting ...")

from gp_retro_repr import (
    Inventory, ReactionTemplateRegistry, ReactionTemplate,
    Program, Select, ApplyTemplate, Stop,
)
from gp_retro_feas import FeasibleExecutor
from gp_retro_obj import (
    ObjectiveSpec,
    RouteFitnessEvaluator,
    epsilon_lexicase_select,
    nsga2_survivor_selection,
)


# -----------------------------------------------------------------------------
# 1. 构造一个最小世界：库存 + 模板 + 可行执行器
# -----------------------------------------------------------------------------
def build_world():
    """
    和 run_feas_demo.py 保持一致的简单设置：
    - 库存：["CC=O", "O"]
    - 目标分子：CCO
    - 模板 T1: C-O -> C=O + O
      这样：应用一次 T1 后，得到的分子都在库存中 => 路线 solved = True
    """
    stock = Inventory(["CC=O", "O"])
    reg = ReactionTemplateRegistry()
    reg.add(
        ReactionTemplate(
            "T1",
            "[C:1]-[O:2]>>[C:1]=O.[O:2]",
            metadata={"family": "oxidation"},
        )
    )
    exe = FeasibleExecutor(reg, inventory=stock)
    target = "CCO"
    return stock, reg, exe, target


# -----------------------------------------------------------------------------
# 2. 定义一个简单的“审计函数”给 RouteFitnessEvaluator 用
# -----------------------------------------------------------------------------
def make_audit_fn(stock: Inventory, target_smiles: str):
    """
    RouteFitnessEvaluator 期望有一个 audit_fn(route)：
    返回：
        is_solved: bool
        first_invalid_molecule_set: List[str]
        current_molecule_set: List[str]
        n_steps: int
        n_valid_steps: int

    这里我们用最简单的规则：
    - 如果有步骤：final_set = 最后一步的 updated_molecule_set
    - 如果没有步骤（例如直接 Stop）：final_set = [target_smiles]
    - is_solved: final_set 中所有分子都在库存中
    - 未解时，first_invalid_molecule_set = final_set
    """
    def audit_route(route):
        # route.steps 是顺序化路线
        if route.steps:
            final_set = route.steps[-1].updated_molecule_set
            n_steps = len(route.steps)
        else:
            # 没有任何 step，视为还停留在 target 分子
            final_set = [target_smiles]
            n_steps = 0

        is_solved = all(stock.is_purchasable(m) for m in final_set)

        return {
            "is_solved": is_solved,
            "first_invalid_molecule_set": [] if is_solved else list(final_set),
            "current_molecule_set": list(final_set),
            "n_steps": n_steps,
            "n_valid_steps": n_steps,  # 这里暂时认为所有已有步都是 valid
        }

    return audit_route


# -----------------------------------------------------------------------------
# 3. 多目标配置
# -----------------------------------------------------------------------------
def build_objectives():
    specs = {
        "solved":            ObjectiveSpec("solved", "max", weight=100.0),
        "route_len":         ObjectiveSpec("route_len", "min", weight=1.0),
        "valid_prefix":      ObjectiveSpec("valid_prefix", "max", weight=1.0),
        "sc_partial_reward": ObjectiveSpec("sc_partial_reward", "max", weight=5.0),
        "purch_frac":        ObjectiveSpec("purch_frac", "max", weight=2.0),
        # 若环境没有 RDKit，qed 会自动被忽略，不影响运行
        "qed":               ObjectiveSpec("qed", "max", weight=1.0),
    }
    return specs


# -----------------------------------------------------------------------------
# 4. 主流程：执行两条 DP 程序 → 计算多目标适应度 → 演示选择算子
# -----------------------------------------------------------------------------
def main():
    print(">>> main() entered in run_obj_demo.py")

    # 4.1 世界与目标
    stock, reg, exe, target = build_world()
    print(f"Target SMILES: {target}")
    print(f"Stock molecules: {list(stock)}")

    # 4.2 定义两条 DP 程序
    # good_route: 先 Select(0) 再 ApplyTemplate("T1")，最后 Stop
    prog_good = Program(
        [
            Select(0),
            ApplyTemplate("T1", rational="good_demo"),
            Stop(),
        ]
    )

    # bad_route: 直接 Stop（意味着“什么都没做”，最后分子集被我们视为 [target]）
    prog_bad = Program(
        [
            Stop(),
        ]
    )

    programs = [
        ("good_route", prog_good),
        ("bad_route", prog_bad),
    ]

    # 4.3 构建多目标适应度评估器
    specs = build_objectives()
    audit_fn = make_audit_fn(stock, target_smiles=target)

    evaluator = RouteFitnessEvaluator(
        objective_specs=specs,
        purchasable_fn=stock.is_purchasable,
        audit_fn=audit_fn,
        target_smiles=target,
    )

    population = []

    # 4.4 执行每个程序，打印路线 JSON 和多目标适应度
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
            {
                "name": name,
                "program": prog,
                "route": route,
                "fitness": fit,
            }
        )

    # 4.5 演示多目标选择算子：ε-lexicase（父代选择）+ NSGA-II（生存者选择）
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


# -----------------------------------------------------------------------------
# 5. 入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception:
        # 确保任何错误都能在控制台看到
        traceback.print_exc()
        sys.exit(1)
