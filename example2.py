from simplex_imp import simplex

"""
showcase the new feature of show_range
"""
c = "max 50+30"
Ab = [
    "1+1 <= 100",
    "2+1 <= 120", 
    "1+2 <= 180"
]

simp = simplex(c, Ab)
simp.show_range("last")