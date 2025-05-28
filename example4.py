from simplex_imp import simplex
"""
show unfeasible Problem
"""
c = "max 1+1"
Ab = [
    "1+1 <= 1",
    "1+1 >= 3"
]

simp = simplex(c, Ab)
simp.show_range()