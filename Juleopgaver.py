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

#%% Opgave 9

# Beregn liste a
from sympy import divisor_count
import multiprocessing


def printdivisors(n):
    d = divisor_count(n)
    if d ==99:
        print(f"{n} {d}")
        return n

pool = multiprocessing.Pool()

a = pool.map(printdivisors, range(1000000000))
pool.close()
pool.join()

a = list(filter(None, a))

#%%
from sympy import divisor_count, primefactors, factorint

a = [230400 ,
451584 ,
12616704 ,
3326976 ,
1115136 ,
6718464 ,
1254400 ,
1557504 ,
4326400 ,
7398400 ,
13542400 ,
2663424 ,
4875264 ,
7750656 ,
3097600 ,
17040384 ,
8479744 ,
20358144 ,
5904900 ,
14500864 ,
11573604 ,
6071296 ,
8856576 ,
25887744 ,
9241600 ,
20939776 ,
15116544 ,
28579716 ,
18113536 ,
15492096 ,
26543104 ,
21529600 ,
24601600 ,
32080896 ,
35046400 ,
47334400 ,
41370624 ,
35808256 ,
44729344 ,
50013184 ,
42198016 ,
48219136 ,
39917124 ,
34292736 ,
43033600 ,
49112064 ,
56550400 ,
46457856 ,
71910400 ,
57517056 ,
72335025 ,
63489024 ,
72999936 ,
62473216 ,
68260644 ,
65545216 ,
68690944 ,
85266756 ,
94012416 ,
91546624 ,
97772544 ,
89113600 ,
95257600 ,
86713344 ,
92775424 ,
84345856 ,
156550144,
240870400,
129049600,
158155776,
186704896,
271590400,
159769600,
104203264,
244859904,
105513984,
273703936,
301925376,
106832896,
136422400,
275825664,
304153600,
109495296,
248882176,
110838784,
249482025,
166306816,
195552256,
140944384,
169624576,
114918400,
225240064,
252937216,
198640836,
226984356,
227165184,
117679104,
310887424,
172975104,
145540096,
119071744,
284394496,
229098496,
257025024,
174662656,
313148416,
202777600,
148644864,
176358400,
204604416,
178062336,
178623225,
261145600,
124947684,
208282624,
290907136,
236913664,
319980544,
210134016,
293094400,
326886400,
382280704,
267387904,
355247104,
295289856,
410305536,
323352324,
439321600,
357663744,
494617600,
412902400,
497468416,

]

b = 1267650600228229401496703205376 # Via OEIS https://oeis.org/A005179

for a_i in a:
    ab = a_i*b
    print(ab)
    print(divisor_count(ab))
    if divisor_count(ab) == 1199:
        print(f"a = {a_i}")
        print(primefactors(ab))
        print(factorint(ab))
        print("-------------------------------")


#%% Opgave 13

from sympy.utilities.iterables import multiset_permutations

ints = range(10)

nums = list(range(100, 1000))

perms = []
for num in nums:
    perms.append(list(multiset_permutations([num // 100 % 10, num // 10 % 10, num // 1 % 10])))

perms = [sorted(perm) for perm in perms]


nodup = []
for perm in perms:
    if perm not in nodup:
        nodup.append(perm)

families = []

for n in nodup:
    number_row = []
    for num in n:
        number = num[0]*100 + num[1]*10 + num[2]
        if number >= 100:
            number_row.append(number)
    families.append(number_row)
        
family_list = []

checks = [1,3,5,7,9,11]

for family in families:
    familycheck = []
    for num in family:
        numcheck = []
        for check in checks:
            numcheck.append((num % check) == 0)
        familycheck.append(numcheck)
    family_list.append(familycheck)

fams = []

for i, family in enumerate(family_list):
    if len(family) > 1:
        family = [any(i) for i in zip(*family)]
    else:
        family = family[0]
    fams.append(family)
    if family == [True,True,True,True,True,True]:
        print(i)
        print(families[i])

#%% Opgave 15



def s(roll):
    return np.sum(((roll[:,0] + roll[:,1]) % 2) == 0)

def m(roll):
    return np.sum(((roll[:,0] * roll[:,1]) % 2) == 0)

def a(roll):
    return np.sum(np.logical_or((roll[:,0] % 2) == 1, (roll[:,1] % 2) == 1))

    
diceroll = np.random.randint(1, 6+1, (400, 2))

s1 = s(diceroll)
m1 = m(diceroll)
a1 = a(diceroll)

i = 0
while s1 != 188 or m1 != 295:
    diceroll = np.random.randint(1, 6+1, (400, 2))
    
    s1 = s(diceroll)
    m1 = m(diceroll)
    a1 = a(diceroll)
    
    print(f"s={s1}, m={m1}, a={a1}")
    i += 1

print(i)

