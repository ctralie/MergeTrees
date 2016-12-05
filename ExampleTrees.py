from MergeTree import *
import numpy as np
import matplotlib.pyplot as plt

def getTreeA():
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([0, 8]))
    A = MergeNode(np.array([0, 5]))
    T.root.addChild(A)
    B = MergeNode(np.array([-3, 0]))
    C = MergeNode(np.array([1, 4]))
    A.addChildren([B, C])
    D = MergeNode(np.array([0.5, 3.5]))
    E = MergeNode(np.array([2, 2]))
    C.addChildren([D, E])
    return T

def getTreeB():
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([0, 8]))
    A = MergeNode(np.array([0, 5]))
    T.root.addChild(A)
    B = MergeNode(np.array([-1, 4]))
    C = MergeNode(np.array([2, 2]))
    A.addChildren([B, C])
    D = MergeNode(np.array([-2, 0]))
    E = MergeNode(np.array([-0.5, 3.5]))
    B.addChildren([D, E])
    return T

if __name__ == "__main__":
    TA = getTreeA()
    TA.addOffset(np.array([0, 0.3]))
    TB = getTreeB()
    offsetA = np.array([0, 0])
    offsetB = np.array([6, 0])
    debug = DebugOffsets(offsetA, offsetB)

    C = doBruteForceMap(TA, TB)
    drawMap(C, offsetA, offsetB)
    plt.show()
