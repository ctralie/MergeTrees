"""
Programmer: Chris Tralie
Purpose: To Implement Laplacian Curve Editing to make smooth deformations
to time series
https://ensiwiki.ensimag.fr/index.php/Alexandre_Ribard_:_Laplacian_Curve_Editing_--_Detail_Preservation
"""
import numpy as np
import scipy.linalg
from scipy import sparse
import scipy.sparse.linalg as slinalg    
from scipy.sparse.linalg import lsqr
    
def doLaplacianWarp(x, anchorsIdx, anchors, anchorWeights, weighted = True):
    #Step 1: Create laplacian matrix
    N = len(x) #Number of vertices
    M = N - 1 #Number of edges
    I = np.zeros(M*2)
    J = np.zeros(M*2)
    
    Y = np.zeros((N, 2))
    Y[:, 0] = np.arange(N)*(np.max(x) - np.min(x))/N
    Y[:, 1] = x
    
    V = np.ones(M*2)
    weighted = False
    if weighted:
        Ds = np.sqrt(np.sum((Y[1:, :] - Y[0:-1, :])**2, 1))
        V = np.concatenate((Ds[:, None], Ds[:, None]), 0).flatten()
    I[0:M] = np.arange(0, N-1)
    J[0:M] = np.arange(1, N)
    I[M:2*M] = np.arange(1, N)
    J[M:2*M] = np.arange(0, N-1)
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    W = np.array(L.sum(1)).flatten()
    L = sparse.dia_matrix((W, 0), L.shape) - L


    coo = L.tocoo()
    NAnchors = len(anchorsIdx)
    [I, J, V] = [coo.row.tolist(), coo.col.tolist(), coo.data.tolist()]
    I = I + (N + np.arange(NAnchors)).tolist()
    J = J + anchorsIdx.tolist()
    V = V + anchorWeights.tolist()
    [I, J, V] = [np.array(I), np.array(J), np.array(V)]
    L = sparse.coo_matrix((V, (I, J)), shape=(N+NAnchors, N)).tocsr()
    deltaCoords = L.dot(Y)
    deltaCoords[N::, 0] = Y[anchorsIdx, 0]
    deltaCoords[N::, 1] = anchors
    
    ret = np.zeros((N, 2))
    for i in range(2):
        ret[:, i] = lsqr(L, deltaCoords[:, i])[0]

    return ret[:, 1]


