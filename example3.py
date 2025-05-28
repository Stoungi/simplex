
from simplex_imp import simplex
"""
show degeneracy
"""
c = "max 3+2" 
Ab = [
    "1+1 <= 4",
    "1+0 <= 2",
    "0+1 <= 2",
    "1+1 >= 4"
]

simp = simplex(c, Ab)
simp.show_range()
