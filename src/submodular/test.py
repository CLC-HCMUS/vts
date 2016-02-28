__author__ = 'MichaelLe'

import copy
import numpy as np
import submodular as sm

a = np.array([1,1,3],ndmin=2)
b = np.array([10,4,3],ndmin=2)
c = np.array([1,1,1],ndmin=2)
d1 = []
d1.append(a)
d1.append(b)
d1.append(c)

a = np.array([2,10,3],ndmin=2)
b = np.array([1,1,1],ndmin=2)
c = np.array([1,5,1],ndmin=2)
d2 = []
d2.append(a)
d2.append(b)
d2.append(c)

P = []
P.append(d1)
P.append(d2)

print P

#create simM
V = []
new_P = []
number_of_word_V = []
k = 0
for p in P:
    new_p = []
    for s in p:
        new_p.append(k)
        k +=1
        V.append(s)
    new_P.append(new_p)
V_word = np.array([1, 5, 3, 7, 3, 7])

n = np.size(V,axis = 0)
max_word = 10
alpha = 0.2
lamda = 0.8

print np.sort(sm.SubmodularFunc(V,n,new_P,V_word,alpha,lamda,max_word))
