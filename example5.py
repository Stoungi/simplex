from simplex_imp import simplex
"""
show unbounded
"""
c = "max 1+1"
Ab = [
    "1-1 >= 0"
]


simp = simplex(c, Ab)
simp.show_range()