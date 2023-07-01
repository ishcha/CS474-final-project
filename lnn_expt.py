from lnn import *

x,y = Variables("x","y")
R1 = Predicate("R1", 2)
R2 = Predicate("R2", 2)
Res = Predicate("Res", 2, world=World.OPEN)
m = Model()
T01 = And(R2(x,y), R1(x,y))
T1 = Implies(Res(x,y), T01)  # And(Implies(And(R1(x,y),R2(x,y)), Res(x,y)), Implies(Res(x,y), And(R1(x,y),R2(x,y))))
# T1.add_data({
#     ("x1","y1"): Fact.TRUE,
#     ("x2","y2"): Fact.TRUE
# })
T2 = Implies(T01, Res(x,y))  # And(Implies(And(R1(x,y),R2(x,y)), Res(x,y)), Implies(Res(x,y), And(R1(x,y),R2(x,y))))
# T2.add_data({
#     ("x1","y1"): Fact.TRUE,
#     ("x2","y2"): Fact.TRUE
# })
T3 = Iff(T01, Res(x,y)) # And(T1, T2)
T3.add_data({
    ("x1","y1"): Fact.TRUE,
    ("x2","y2"): Fact.TRUE
})
# T1.add_data(Fact.TRUE)
m.add_knowledge(T3) #01, T1, T2, T3)
m.add_data({
    R1: {("x1","y1"): Fact.TRUE,
    ("x2", "y2"): Fact.FALSE},
    R2: {
    ("x1", "y1"): Fact.FALSE,
    ("x2", "y2"): Fact.TRUE},
    Res: {("x1", "y1"): Fact.TRUE,
    ("x2", "y2"): Fact.FALSE
    }
})
# R1.add_data({
#     ("x1","y1"): Fact.TRUE,
#     ("x2", "y2"): Fact.TRUE
# })
# R2.add_data({
#     ("x1", "y1"): Fact.FALSE,
#     ("x2", "y2"): Fact.TRUE,
# })
# Res.add_data({
#     ("x1", "y1"): Fact.TRUE,
#     ("x2", "y2"): Fact.FALSE
# })
m.train(losses = [Loss.CONTRADICTION], direction=Direction.DOWNWARD)

m.add_data({
    # R1: {("x3", "y3"): Fact.FALSE},
    # R2: {("x3", "y3"): Fact.TRUE},
    R1: {("x4", "y4"): Fact.TRUE},
    R2: {("x4", "y4"): Fact.FALSE},
    Res: {("x4","y4"): Fact.UNKNOWN}
})
T3.add_data({
    ("x4","y4"): Fact.TRUE
})
m.infer()
# m.infer(direction=Direction.DOWNWARD)
Res.print()
# T01.print(params=True)
# T1.print(params=True)
# T2.print(params=True)
# T3.print(params=True)
# model.train(losses = [loss.CONTRADICTION])
m.print()
m.plot_graph()
# formulae = [
#     Smoking_causes_Cancer
#     Smokers_befriend_Smokers
# ]
# model.add_knowledge(*formulae, world=World.AXIOM)
