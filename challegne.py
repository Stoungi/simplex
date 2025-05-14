
from simplex_imp import simplex




"""
    z = 1x + 2y

    x <= 5



"""

c = "max 1+2"  # objective function    

Ab = ["1+0 <= 5"] # 1st constraint
     



simp = simplex(c, Ab)

simp.show_steps()
print("Optimal solution:", simp.optimal)
print("Maximum value:", simp.value)

print("steps taken: ", simp.steps)
