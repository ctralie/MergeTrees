import numpy as np
import matplotlib.pyplot as plt

def polyFit(X, xs):
    """
    Given a Nx2 array X of 2D coordinates, fit an N^th order polynomial
    and evaluate it at the coordinates in xs.
    This function assumes that all of the points have a unique X position
    """
    x = X[:, 0]
    y = X[:, 1]
    N = X.shape[0]
    A = np.zeros((N, N))
    for i in range(N):
        A[:, i] = x**i
    AInv = np.linalg.inv(A)
    b = AInv.dot(y[:, None])

    M = xs.size
    Y = np.zeros((M, 2))
    Y[:, 0] = xs
    for i in range(N):
        Y[:, 1] += b[i]*(xs**i)
    plt.plot(Y[:, 0], Y[:, 1], 'b')
    plt.hold(True)
    plt.scatter(X[:, 0], X[:, 1], 20, 'r')
    plt.show()
    return Y
