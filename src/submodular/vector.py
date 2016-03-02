__author__ = 'MichaelLe'

from numpy import *
from numpy import linalg as LA


def converArr(s):
    lenS = ceil(len(s)/2.0)
    a = zeros((1,lenS ))
    i=0
    for c in s:
        if (c != ' '):
            a[0,i] = c
            i = i+1
    return a

def dotProduct(a, b):
    n = size(a,0)
    sum = 0
    for i in range(0,n):
        sum = sum + a[i]*b[i]
    return sum

def cosine(a, b):
    c =  dotProduct(a,b)
    d =  linalg.norm(a)*linalg.norm(b)
    return (c/d + 1)/2

def similarity(a,b):
    re = cosine(a,b)
    return re
