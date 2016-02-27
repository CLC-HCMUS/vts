__author__ = 'MichaelLe'


import string
import numpy
import vector


def loadfile(filename):
    f = open(filename,'r')
    a = []
    for line in f:
        s = (str(line))
        s = s.replace('\n','')
        a.append(vector.converArr(s))
    return a






