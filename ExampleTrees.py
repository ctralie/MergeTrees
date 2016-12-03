from MergeTree import *
import numpy as np
import matplotlib.pyplot as plt

def getTreeA():
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(8, np.array([0, 8]))
    A = MergeNode(5, np.array([0, 5]))
    T.root.addChild(A)
    B = MergeNode(0, np.array([-3, 0]))
    C = MergeNode(4, np.array([1, 4]))
    A.addChildren([B, C])
    D = MergeNode(0.5, np.array([0.5, 3.5]))
    E = MergeNode(2, np.array([2, 2]))
    C.addChildren([D, E])
    return T

if __name__ == "__main__":
    T = getTreeA()
    T.render(np.array([0, 0]))
    plt.show()