__author__ = 'MichaelLe'


import numpy as np
import vector as vec
from numpy import linalg
import mmrelevance as mmr

a = np.array([10,4,10])
b = np.array([1,3,13])
c = []

c.append(a)
c.append(b)

len_sen = np.array([3,3])

print (linalg.norm(a))

print mmr.summaryMMR12(c, len_sen,0.3, 5, 1)