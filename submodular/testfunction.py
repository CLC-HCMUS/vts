__author__ = 'HyNguyen'
import numpy as np
import os
import sys

print("hyh",sys.path[0])
print(os.path.abspath(__file__))

def cosine(a, b):
    c =  np.dot(a,b)
    d =  np.linalg.linalg.norm(a)*np.linalg.linalg.norm(b)
    return (c/d + 1)/2

if __name__ == "__main__":

    X = np.array([[0,1,2],
                  [3,2,3],
                  [5,0,0]],dtype=float)
    matrix = np.zeros_like(X,dtype=np.float32)
    for i in range(X.shape[0]):
        for j in range(i+1,X.shape[0]):
            matrix[i][j] = cosine(X[i],X[j])
            print(cosine(X[i],X[j]))
    print matrix
    xmin = np.min(X)
    xmax = np.max(X)
    C = (X - xmin)/(xmax - xmin)
    print C