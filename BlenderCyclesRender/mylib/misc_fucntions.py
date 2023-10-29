import os
import torch
import numpy as np
from scipy.linalg import logm, expm

def ManualBoundingBox(a_org):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    output = torch.tensor(()).to(device)
    for i in range(a_org.size(0)):
        a = a_org[i,:,:]
        w_min = min(torch.where(a == 0)[1])
        h_min = min(torch.where(a == 0)[0])
        w_max = max(torch.where(a == 0)[1])
        h_max = max(torch.where(a == 0)[0])

        output = torch.cat((output,torch.Tensor([[w_min,h_min,w_max, h_max]]).to(device)), 0)  
    return output

def Hom2twist(H):
    # HomogMatrix2twist Convert 4x4 homogeneous matrix to twist coordinates
    '''
    Input:  H(4,4): Euclidean transformation matrix (rigid body motion)
    Output:
        twist(6,1): twist coordinates. Stack linear and angular parts [v;w]
        Observe that the same H might be represented by different twist vectors
        Here, twist(4:6) is a rotation vector with norm in [0,pi]
    '''
    se_matrix = logm(H)

    # careful for rotations of pi; the top 3x3 submatrix of the returned
    # se_matrix by logm is not skew-symmetric (bad).

    v = se_matrix[0:3,3]
    w = Matrix2Cross(se_matrix[0:3,0:3])

    twist = np.array([v[0],v[1],v[2],w[0],w[1],w[2]])

    return twist

def twist2Hom(twist):
    #twist2HomogMatrix Convert twist coordinates to 4x4 homogeneous matrix
    '''
     Input:
        -twist(6,1): twist coordinates. Stack linear and angular parts [v;w]
     Output:
        -H(4,4): Euclidean transformation matrix (rigid body motion)
    '''
    v = twist[0:3] #linear part
    v = np.array([[v[0]],[v[1]],[v[2]]])
    w = twist[3:6] #angular part

    se_matrix = np.concatenate((np.concatenate((Cross2Matrix(w), v), axis=1),np.zeros((1, 4))),axis=0) # Lie algebra matrix

    H = expm(se_matrix)

    return H

def RT2hom(R,T):
    H = np.identity(4)
    H[0:3,0:3] = R

    H[0,3] = T[0]
    H[1,3] = T[1]
    H[2,3] = T[2]

    return H
    
def RT2homRow(R,T):
    H = np.identity(4)
    H[0:3,0:3] = R

    H[0,3] = T[0]
    H[1,3] = T[1]
    H[2,3] = T[2]

    return np.expand_dims( np.reshape(H[0:3,0:4], 12), axis=0)

def Cross2Matrix(s):

    S = np.zeros((3,3))

    S[0,1]=-s[2]
    S[0,2]=s[1]
    S[1,0]=s[2]
    S[1,2]=-s[0]
    S[2,0]=-s[1]
    S[2,1]=s[0]

    return S
    
def Matrix2Cross(S):
    return [-S[1,2], S[0,2], -S[0,1]]
    
#def H_AB_to_H_BA(H_AB):  
#    # A->B  to B->A
#    # relative to camera frame
#    # Pc = Rw^TPw + (-Rw^T Tw)
#    # Hc = [Rw^T | -Rw^T Tw] = [Rc | Tc]
#    H_BA = np.zeros((4,4))
#    H_BA[0:3,0:3] = H_AB[0:3,0:3].T
#    H_BA[0:3,3] = -H_AB[0:3,0:3].T @ H_AB[0:3,3]
#    H_BA[3,3] = 1
#    return H_BA
