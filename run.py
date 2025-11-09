# run.py
from gp_retro_repr import (
    Inventory, ReactionTemplateRegistry, ReactionTemplate,
    Program, Select, ApplyTemplate, Stop, ExecutionConfig
)

def main():
    # 1) 可购库存（示例）  定义的测试：库存数据库就是这样
    stock = Inventory(["CCO", "O=C=O"])

    # 2) 模板库（示例 retro 模板）
    reg = ReactionTemplateRegistry()
    reg.add(ReactionTemplate("T1", "[C:1]-[O:2]>>[C:1]=O.[O:2]"))

    # 3) DP 程序
    prog = Program([Select(0), ApplyTemplate("T1", rational="disconnection"), Stop()])  #自己定义的名称（模板名称）
    cfg = ExecutionConfig(template_registry=reg, inventory=stock)
    #####DP：相当于一个指令集，告诉解释器我要怎么做逆合成。
         #DP = 用一小段“程序”把“如何一步步把目标分子拆成可购原料”描述出来。

    # 4) 执行程序
    route = prog.execute(target_smiles="CCO", config=cfg)
    print(route.to_json())
    print(route.is_solved(stock))  # 可能是 False（示例库存很小）

if __name__ == "__main__":
    try:
        main()
    except Exception as e:        
        import traceback; traceback.print_exc()

#就是：我有一个自定义库存数据库，里面存放的是可购分子；
# 有一个自定义的 retro 模板库，里面存放的是一些 retro 模板；
#我定义了一个 Decision Program，告诉解释器我要怎么做 retrosynthesis。
#     而这个DP也是按照我定义好的模板来执行，执行的操作名称也是自己定义 rational。

#z这个DP与我要实现的研究任务 GP之间的关系：
   #理解DP是基因 的编码方式（基因型）；GP所要考虑的就是如何搜索/进化这些程序（基因）的方法；
     #两者相互配合，GP 在程序空间里进化 DP；DP 被解释器执行后产出一条具体的逆合成路线（表型）。
# DP：问题表示/怎么做逆合成；
# Interpreter(executor)：把DP变成顺序化路线（表型）
# GP : 在DP空间里搜索/进化好的DP
