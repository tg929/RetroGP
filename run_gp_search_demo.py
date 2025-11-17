# run_gp_search.py
# 一个“最小可跑”的多目标 GP 搜索主循环
# - 初始化随机 DP 程序种群
# - 评估(执行->多目标->标量)
# - 选择(ε-lexicase父代 + NSGA-II生存者)
# - 交叉/突变
# - 多代进化，输出 Pareto 方向上的优秀路线

import random
import copy
import statistics
import sys
import traceback
from typing import List, Dict, Any, Tuple

# 表示层 / 可行性层
from gp_retro_repr import Program, Route, Select, ApplyTemplate, Stop
from gp_retro_feas import FeasibleExecutor

# 目标与适应度层
from gp_retro_obj import (
    RouteFitnessEvaluator,
    epsilon_lexicase_select,
    nsga2_survivor_selection,
)

# 统一的“世界配置”和“目标配置”
from demo_utils import build_world_t1, build_objectives_default


# --------------------------------------------------------------------
# 0) 审计函数：把 Route → is_solved / 当前分子集 / 步数 等信息
# --------------------------------------------------------------------
def make_audit_fn(stock, target_smiles: str):
    """
    RouteFitnessEvaluator 期望有一个 audit_fn(route)，返回：
        is_solved: bool
        first_invalid_molecule_set: List[str]
        current_molecule_set: List[str]
        n_steps: int
        n_valid_steps: int
    这里给一个和 run_obj_demo.py 一致的简单实现。
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


# --------------------------------------------------------------------
# 1) DP 程序编码与操作（模板序列 <-> Program）
# --------------------------------------------------------------------
def templates_of_program(prog: Program) -> List[str]:
    """抽取 Program 中所有 ApplyTemplate 的 template_id 顺序列表"""
    tids = []
    for instr in prog.instructions:
        if isinstance(instr, ApplyTemplate):
            tids.append(instr.template_id)
    return tids


def program_from_templates(template_ids: List[str]) -> Program:
    """把 template 序列还原成标准 DP：Select(0)->Apply(T)->Select(0)->...->Stop()"""
    steps = []
    for tid in template_ids:
        steps.append(Select(0))                  # 每次应用前选中目标分子（简单示例：总是 index=0）
        steps.append(ApplyTemplate(tid, rational="gp"))
    steps.append(Stop())
    return Program(steps)


def random_program(template_pool: List[str], min_len=0, max_len=3) -> Program:
    k = random.randint(min_len, max_len)
    tids = [random.choice(template_pool) for _ in range(k)]
    return program_from_templates(tids)


def crossover_one_point(p1: Program, p2: Program) -> Tuple[Program, Program]:
    """单点交叉：在模板序列上做切分再拼接"""
    t1 = templates_of_program(p1)
    t2 = templates_of_program(p2)
    c1 = random.randint(0, len(t1))
    c2 = random.randint(0, len(t2))
    child1 = program_from_templates(t1[:c1] + t2[c2:])
    child2 = program_from_templates(t2[:c2] + t1[c1:])
    return child1, child2


def mutate_program(
    p: Program,
    template_pool: List[str],
    p_insert=0.40,
    p_delete=0.30,
    p_modify=0.30,
    max_total_len=5,
) -> Program:
    """在模板序列上做插入/删除/替换"""
    t = templates_of_program(p)
    op = random.random()
    if op < p_insert:
        # 插入
        if len(t) < max_total_len:
            pos = random.randint(0, len(t))
            t.insert(pos, random.choice(template_pool))
    elif op < p_insert + p_delete:
        # 删除
        if len(t) > 0:
            pos = random.randrange(len(t))
            t.pop(pos)
    else:
        # 修改
        if len(t) > 0:
            pos = random.randrange(len(t))
            t[pos] = random.choice(template_pool)
    return program_from_templates(t)


# --------------------------------------------------------------------
# 2) 个体评估：Program -> Route -> Fitness
# --------------------------------------------------------------------
def evaluate_program(
    prog: Program,
    exe: FeasibleExecutor,
    evaluator: RouteFitnessEvaluator,
    target: str,
) -> Dict[str, Any]:
    # Executor may fail if a template has no applicable reactants (e.g., T1 on a non-alcohol).
    # Treat that as an infeasible route instead of crashing the whole GP run.
    error_reason = None
    try:
        route = exe.execute(prog, target_smiles=target)
    except Exception as e:
        route = Route()  # empty route => audit_fn marks as unsolved
        error_reason = str(e)

    fit = evaluator.evaluate(route)
    if error_reason:
        fit.extra["executor_error"] = error_reason
    return {"program": prog, "route": route, "fitness": fit}


# --------------------------------------------------------------------
# 3) GP 主循环
# --------------------------------------------------------------------
def run_gp_search(
    pop_size=20,
    generations=15,
    p_crossover=0.7,
    p_mutation=0.4,
    seed=123,
):
    random.seed(seed)

    # 世界 + 评估器（都来自 demo_utils）
    stock, reg, target = build_world_t1()
    exe = FeasibleExecutor(reg, inventory=stock)

    specs = build_objectives_default()
    evaluator = RouteFitnessEvaluator(
        objective_specs=specs,
        purchasable_fn=stock.is_purchasable,
        audit_fn=make_audit_fn(stock, target_smiles=target),
        target_smiles=target,
    )
    senses = {k: spec.direction() for k, spec in specs.items()}
    objective_keys = list(specs.keys())

    template_pool = list(reg.templates.keys())  # 例如 ["T1", ...]

    # ------ 初始化种群 ------
    population: List[Dict[str, Any]] = []
    for _ in range(pop_size):
        prog = random_program(template_pool, min_len=0, max_len=3)
        population.append(evaluate_program(prog, exe, evaluator, target))

    # ------ 进化循环 ------
    for gen in range(1, generations + 1):
        scalars = [ind["fitness"].scalar for ind in population]
        solved_count = sum(
            1 for ind in population if ind["fitness"].objectives.get("solved", 0) > 0.5
        )
        best = max(population, key=lambda ind: ind["fitness"].scalar)

        print(f"\n=== Gen {gen}/{generations} ===")
        print(
            f"  solved: {solved_count}/{len(population)}  "
            f"best_scalar: {best['fitness'].scalar:.3f}  "
            f"mean_scalar: {statistics.mean(scalars):.3f}"
        )

        # --- 父代选择：ε-lexicase ---
        parents = epsilon_lexicase_select(
            population,
            senses=senses,
            objective_keys=objective_keys,
            eps_quantile=0.10,
            n_parents=pop_size,
        )

        # --- 生成子代：交叉 + 突变 + 评估 ---
        offspring: List[Dict[str, Any]] = []
        i = 0
        while len(offspring) < pop_size:
            # 交叉或复制
            if random.random() < p_crossover and i + 1 < len(parents):
                p1 = parents[i]["program"]
                p2 = parents[i + 1]["program"]
                c1, c2 = crossover_one_point(p1, p2)
                i += 2
                children = [c1, c2]
            else:
                p0 = parents[i % len(parents)]["program"]
                children = [copy.deepcopy(p0)]
                i += 1

            # 突变
            new_children = []
            for ch in children:
                if random.random() < p_mutation:
                    ch = mutate_program(ch, template_pool)
                new_children.append(ch)

            # 评估
            for ch in new_children:
                offspring.append(evaluate_program(ch, exe, evaluator, target))
                if len(offspring) >= pop_size:
                    break

        # --- 生存者选择：父代 + 子代 → NSGA-II 选出下一代 ---
        combined = population + offspring
        population = nsga2_survivor_selection(
            combined, k=pop_size, senses=senses, objective_keys=objective_keys
        )

    # ------ 结束：输出若干条最优解 ------
    population.sort(key=lambda ind: ind["fitness"].scalar, reverse=True)
    print("\n=== Final top solutions ===")
    topk = min(5, len(population))
    for i in range(topk):
        ind = population[i]
        objs = ind["fitness"].objectives
        print(
            f"[{i+1}] scalar={ind['fitness'].scalar:.3f} "
            f"solved={bool(objs.get('solved', 0))} "
            f"route_len={objs.get('route_len', -1)}"
        )
        print(ind["route"].to_json())


def main():
    run_gp_search(
        pop_size=20,
        generations=15,
        p_crossover=0.7,
        p_mutation=0.4,
        seed=123,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
