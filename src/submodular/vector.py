__author__ = 'MichaelLe'

from numpy import *
from numpy import linalg as LA


def dotProduct(a, b):
    n = size(a,1)
    sum = 0
    for i in range(0,n):
        sum = sum + a[0,i]*b[0,i]
    return sum

def cosine(a, b):
    c = dotProduct(a,b)
    d = linalg.norm(a)*linalg.norm(b)
    return (c/d + 1)/2

def similarity(a,b):
    re = cosine(a,b)
    return re
