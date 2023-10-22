import numpy as np

def fourPointsOnLine(P_A, P_B):
    P_C = P_A + (P_B-P_A) / 3
    P_D = P_A + 2* (P_B-P_A) / 3
    P = np.concatenate((P_C, P_D), axis=0).reshape(2,P_A.shape[0])
    P_all = np.concatenate((P_A, P_C, P_D, P_B), axis=0).reshape(4,P_A.shape[0])
    return P_all, P
