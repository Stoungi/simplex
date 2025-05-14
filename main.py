
from simplex_imp import simplex



"""
z = 50x + 30y

4x + 3y <= 240

y <= 40

x <= 36
"""

     

     
c = "min -6+7+4"
Ab = ["2+5-1 <= 18", 
      "1-1-2 <= -14",
      "3+2+2 = 26"]

c2 = "max 50+30" 

Ab2 = ["4+3 <= 240",
      "0+1 <= 40",
      "1+0 <= 36"] 
      
     

simp = [simplex(c, Ab), simplex(c2, Ab2)]
for a in simp:
      a.show_steps()
      print("optimal: ", a.optimal)
      print("value: ", a.value)
      print("==================")


