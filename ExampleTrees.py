from MergeTree import *
from BruteForceMap import *
from ZSSMap import *
import numpy as np
import matplotlib.pyplot as plt

def getTreeA():
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([0, 5]))
    B = MergeNode(np.array([-6, 0]))
    C = MergeNode(np.array([8, 2]))
    T.root.addChildren([B, C])
    return T

def getTreeAA():
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([0, 8]))
    A = MergeNode(np.array([0, 5]))
    T.root.addChild(A)
    B = MergeNode(np.array([-2, 4]))
    C = MergeNode(np.array([1, 4]))
    A.addChildren([B, C])
    return T

def getTreeB():
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([0, 5]))
    B = MergeNode(np.array([-4, 4]))
    C = MergeNode(np.array([8, 1.5]))
    T.root.addChildren([B, C])
    D = MergeNode(np.array([-10, 0]))
    E = MergeNode(np.array([-0.5, 3]))
    B.addChildren([D, E])
    return T

def getTreeC():
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([0, 10]))
    A = MergeNode(np.array([0, 7]))
    T.root.addChild(A)
    B = MergeNode(np.array([-4, 0]))
    C = MergeNode(np.array([2, 3]))
    A.addChildren([B, C])
    return T

def getTreeD():
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([0, 10]))
    A = MergeNode(np.array([0, 7]))
    T.root.addChild(A)
    B = MergeNode(np.array([-4, 0]))
    C = MergeNode(np.array([1, 5]))
    A.addChildren([B, C])
    D = MergeNode(np.array([0.5, 4]))
    E = MergeNode(np.array([2, 3]))
    C.addChildren([D, E])
    return T

#E/F: Simplest example pair I can come up with that has exactly
#one subdivided node per tree
def getTreeE():
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([0, 5]))
    A = MergeNode(np.array([-2, 3]))
    B = MergeNode(np.array([1, 4]))
    T.root.addChildren([A, B])
    C = MergeNode(np.array([0.5, 3]))
    D = MergeNode(np.array([2, 3]))
    B.addChildren([C, D])
    return T

def getTreeF():
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([0, 5]))
    A = MergeNode(np.array([-1, 4]))
    B = MergeNode(np.array([2, 3]))
    T.root.addChildren([A, B])
    C = MergeNode(np.array([-2, 3]))
    D = MergeNode(np.array([-0.5, 3]))
    A.addChildren([C, D])
    return T

def getPaul1():
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([3, 10]))
    A = MergeNode(np.array([-5, 0]))
    B = MergeNode(np.array([4, 3]))
    T.root.addChildren([A, B])
    return T

def getPaul2():
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([7, 10]))
    A = MergeNode(np.array([6, 3]))
    B = MergeNode(np.array([15, 4]))
    T.root.addChildren([A, B])
    return T


if __name__ == "__main__":
    TA = getTreeA()
    #TA = getPaul1()
    X = TA.getCriticalPtsList()
    xl = np.min(X[:, 0])
    xr = np.max(X[:, 0])

    #TA.addOffset(np.array([0, 0.3]))
    TB = getTreeB()
    #TB = getPaul2()
    offsetA = np.array([0, 0])
    offsetB = np.array([28, 0])
    debug = DebugOffsets(offsetA, offsetB)

    #C = doBruteForceMap(TA, TB)
    C = getZSSMap(TA, TB, True)
    print(C.cost)

    TA.render(offsetA, drawCurved = False)
    TB.render(offsetB, drawCurved = False)

    #drawMap(C, offsetA, offsetB, drawCurved = False, drawSubdivided = True)
    ax = plt.gca()
    yvals = TA.getfValsSorted().tolist() + TB.getfValsSorted().tolist()
    yvals = np.sort(np.unique(np.array(yvals)))
    ax.set_yticks(yvals)
    ax.set_xticks([])
    plt.grid()
    plt.show()
