
from simplex_imp import simplex




"""
    z = 1x + 2y

    x <= 5



"""

c = [1, 2]  # objective function    

A = [[1, 0]] # 1st constraint (left hand side)   
      
     

b = [5] # 1st contraint (right hand side)
     



simp = simplex(c,A,b)

simp.show_steps()
print("Optimal solution:", simp.optimal)
print("Maximum value:", simp.value)

print("steps taken: ", simp.steps)


