# run.py
from gp_retro_repr import (
    Inventory, ReactionTemplateRegistry, ReactionTemplate,
    Program, Select, ApplyTemplate, Stop, ExecutionConfig
)

def main():
    # 1) 可购库存（示例）
    stock = Inventory(["CCO", "O=C=O"])

    # 2) 模板库（示例 retro 模板）
    reg = ReactionTemplateRegistry()
    reg.add(ReactionTemplate("T1", "[C:1]-[O:2]>>[C:1]=O.[O:2]"))

    # 3) DP 程序
    prog = Program([Select(0), ApplyTemplate("T1", rational="disconnection"), Stop()])
    cfg = ExecutionConfig(template_registry=reg, inventory=stock)

    # 4) 执行程序
    route = prog.execute(target_smiles="CCO", config=cfg)
    print(route.to_json())
    print(route.is_solved(stock))  # 可能是 False（示例库存很小）

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 方便你看到错误原因（例如没装 rdchiral / rdkit）
        import traceback; traceback.print_exc()

