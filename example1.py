
from simplex_imp import simplex

"""
showcase the compare method
"""



"""
max z = 50x + 30y

4x + 3y <= 240

y <= 40

x <= 36
"""

c = "max 50+30" 

Ab = ["4+3 <= 240",
      "0+1 <= 40",
      "1+0 <= 36"] 
           
"""
min z = -6x1 + 7x2 + 4x3 

2x1 + 5x2 -1x3 <= 18

1x1 - 1x2 - 2x3 <= -14

3x1 + 2x2 + 2x3 = 26
"""

     
c2 = "min -6+7+4"
Ab2 = ["2+5-1 <= 18", 
      "1-1-2 <= -14",
      "3+2+2 = 26"]


     
# made a list of simplex that has the solutions to both
simp = [simplex(c, Ab), simplex(c2, Ab2)]


simplex.compare(simp)




