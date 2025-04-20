
from simplex_imp import simplex

c1 = [3, 5]  # c1 = objective function    

A1 = [[1, 0], # A1 = constraints (left handed side)           
      [0, 2],            
      [3, 2]]    
     
b1 = [4, 12, 18] # b1 = contraints (right handed side)


# same idea for the 2nd example here
c2 = [1, 2]

A2 = [[1, 0],
      
      [2, 5]]

b2 = [5, 16]



simp = [simplex(c1, A1, b1), simplex(c2, A2, b2)]



for a in simp:

    a.show_range()
    print("Optimal solution:", a.optimal)
    print("Maximum value:", a.value)

    print("steps taken: ", a.steps)




