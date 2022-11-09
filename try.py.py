


class Polynomial:
    # the list a contains the coefficients of the 
    # polynomial starting with a0, a1, ... 
    # p(x) is then a0 + a1*x + a2*x^2 + a3*x^3 + ...
    def __init__(self, a):
        self.a = a
        
    def eval(self, x):
        res = 0.0
        for i in range(0, len(self.a)):
            res += self.a[i] * pow(x,i)
        return res
        
    
    
        
p = Polynomial([1, 2, 1])

print(p.eval(-1))

import math 
print(math.sin(math.pi/2))

            