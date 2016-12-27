from MergeTree import *
from BruteForceMap import *
from ZSSMap import *
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

def getTreeAA():
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([0, 8]))
    A = MergeNode(np.array([0, 5]))
    T.root.addChild(A)
    B = MergeNode(np.array([-3, 4]))
    C = MergeNode(np.array([1, 4]))
    A.addChildren([B, C])
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

if __name__ == "__main__":
    TA = getTreeAA()
    TA.addOffset(np.array([0, 0.3]))
    TB = getTreeAA()
    offsetA = np.array([0, 0])
    offsetB = np.array([6, 0])
    debug = DebugOffsets(offsetA, offsetB)

    #C = doBruteForceMap(TA, TB)
    C = doZSSMap(TA, TB)
    print C.cost


    drawMap(C, offsetA, offsetB, drawSubdivided = True)
    plt.show()
