# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:09:51 2021

@author: nweinr
"""

import numpy

# =============================================================================
# Opgave 1
# =============================================================================

l_max = 10000
w_max = 10000

pairs = []

for l in range(1,l_max+1):
    if l%100 == 0:
        print(l)
    for w in range(1,w_max+1):
        if 4*(l+w)-8 == l*w:
            pairs.append((w,l))