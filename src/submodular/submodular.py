__author__ = 'MichaelLe'

import numpy as np
import vector
import copy
# with S, V is the list of all sentence



def SimMatrix(senList):
    numSen = np.size(senList,0)
    simM = np.ones((numSen + 1, numSen))
    for i in range(numSen):
        for j in range(i,numSen,1):
            simM[i,j] = vector.similarity(senList[i],senList[j])
            simM[j,i] = simM[i,j]
    for i in range(numSen):
        simM[numSen,i] = np.sum(simM[:numSen,i])
    return simM

def countC(v, S, simM):
    #v la so thu tu cau trong V
    #S la tap thu tu cau da tom tat (thu tu tuong ung trong V
    sum_cov = 0
    for c in S:
        sum_cov = sum_cov + simM[v,c]
    return sum_cov


def coverage(S, n, simM, alpha):
    #S: van ban tom tat (chi chua thu tu cau
    #V: tap cau dau vao (chi chua thu tu
    #simM: ma tran similarity
    #n: kich thuoc V
    #alpha: he so trade-off
    sum_cov = 0
    for c in range(n):
        CS = countC(c,S,simM)
        CV = simM[n,c]
        sum_cov = sum_cov + min(CS, alpha*CV)
    return sum_cov

def intersectionSet(a,b):
    #giao 2 cluster a, b
    re = []
    if len(a) >= len(b):
        for t in a:
            if (t in b) == True:
                re.append(t)
        return re
    else:
        return intersectionSet(b,a)


def diversityEachPart(S,Pi,n, simM):
    #S: van ban tom tat
    #Pi: cluster thu i
    #n: so luong cau dau vao V
    #simM: ma tran tuong quan
    A = intersectionSet(S,Pi)
    sum_div = 0
    for a in A:
        sum_div = sum_div + simM[n,a]
    return sum_div

def diversity(S,n,P,simM):
    sum_div = 0
    for p in P:
        sum_div = np.sqrt((1.0/n)*diversityEachPart(S,p, n, simM)) + sum_div
    return sum_div

def f1(S, n, P, simM, alpha, lamda):
    return coverage(S,n,simM,alpha) + lamda*diversity(S,n,P,simM)

def isStopCon(S,number_of_word_V, max_word):
    epsilon = 2
    #count sum word of S:
    sum_S = np.sum(number_of_word_V[S])
    if (sum_S > max_word):
        return 1
    else: return 0

def SubmodularFunc(V,n, P, V_word, alpha, lamda, max_word):
    simM = SimMatrix(V)

    #create V_number
    V_number = range(n)

    #find S
    S = []

    while (isStopCon(S,V_word,max_word)== 0):
        score_matrix = np.zeros(n)
        for i in range(n):
            if (i in S) == False:
                tmp_s = copy.deepcopy(S)
                tmp_s.append(i)
                k = f1(tmp_s,n,P,simM,alpha, lamda)
                score_matrix[i] = k

        selected_sen = np.argmax(score_matrix)
        S.append(selected_sen)

    return S
