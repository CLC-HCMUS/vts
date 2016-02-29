__author__ = 'MichaelLe'

from numpy import *
import vector
# with S, V is the list of all sentence



def C(v, S):
    sum = 0
    for c in S:
        #print v
        #print c
        #print vector.similarity(v,c)                       
        sum = sum + vector.similarity(v,c)
    return sum


def coverage(S, V, alpha):
    sum = 0
    for c in V:
        CS = C(c,S)
        CV = C(c,V)
        sum = sum + min(CS, alpha*CV)
    return sum

def inAList(t, a):
    for s in a:
        if (array_equal(s,t)):
            return 1
    return 0

def intersectionArray(a,b):
    re = []
    if (size(a,0) > size(b,0)):
        for t in a:
            if (inAList(t,b)):
                re.append(t)
    else:
        for t in b:
            if (inAList(t,a)):
                re.append(t)
    return re

def diversityEachPart(S,Pi,V):
    A = intersectionArray(S,Pi)
    sum = 0
    for a in A:
        sum = sum + C(a,V)
    return sum

def diversity(S,V,P):
    sum = 0
    N = size(V,0)
    for p in P:
        sum = sqrt((1.0/N)*diversityEachPart(S,p, V)) + sum
    return sum

def mergePintoV(P):
    V = []
    for p in P:
        for s in p:
            V.append(s)
    return V

def F(S, V, P,alpha,lamda):
    return coverage(S,V,alpha) + lamda*diversity(S,V,P)

def copyMatrix(S):
    V = []
    for s in S:
        V.append(s)
    return V

def notin(e, se):

    for s in se:
        #print s, ' ', e, ' ', se
        if (e == s):
            return 1
    return 0

def addElement(S,V,P,n, alpha, lamda, se):
    max = 0
    i=0
    k = 0
    for i in arange(0,n):
        if (notin(i,se) == 0):
            Temp = copyMatrix(S)
            Temp.append(V[i])
            tempF = F(Temp,V,P, alpha,lamda)
            #print i
            if  tempF > max:
                max = tempF
                k = i
    return k

def maximizeF(V, P, alpha, lamda, L , k):
    S = []
    se = []
    n = size(V,0)
    price = 0
    while (1 == 1):
        e = addElement(S,V, P,n,alpha, lamda, se)
        #print e
        if (price + L[e] < k):
            S.append(V[e])
            se.append(e)
            price = price + L[e]
        else:
            return se
