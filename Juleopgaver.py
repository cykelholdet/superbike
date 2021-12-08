# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:09:51 2021

@author: nweinr
"""

import numpy as np

#%% Opgave 1

l_max = 10000
w_max = 10000

pairs = []

for l in range(1,l_max+1):
    if l%100 == 0:
        print(l)
    for w in range(1,w_max+1):
        if 4*(l+w)-8 == l*w:
            pairs.append((w,l))
            
#%% Opgave 3

n_max = 10000000000


solutions = []
for n in range(1,n_max+1):
    if (n**2+2021)%(n+2021) == 0 and (n**2+2022)%(n+2021) == 0:
        solutions.append(n)
        print(n)

#%% Opgave 6

n_max = 1000000000

solutions = []
for n in range(n_max):
    if np.sqrt(n**2+18*n+2)%1 == 0:
        print(n)
        solutions.append(n)
    if np.sqrt(n**2-18*n+2)%1 == 0:
        print(-n)
        solutions.append(-n)

#%% Opgave 7    

def poly(coefs, x):
    order = len(coefs)
    SUM = 0
    for i in range(order):
        SUM += coefs[i]*x**i
    
    return SUM
    
def isprime(num):
    for n in range(2,int(num**1/2)+1):
        if num%n==0:
            return False
    return True
    

n_max = 100

poly_maxes = [100,100,100,100,100]

for n in range(1,n_max+1):
    for a in range(poly_maxes[0]):
        for b in range(poly_maxes[1]):
            for c in range(poly_maxes[2]):
                for d in range(poly_maxes[3]):
                    for e in range(poly_maxes[4]):
                        if poly([a,b,c,d,e], 4) == 0 and poly([a,b,c,d,e], 5) == 0:
                            if isprime(poly([a,b,c,d,e], n)):
                                print(n)











