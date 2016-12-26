#Wrap around the ZSS library for Zhang/Shasha
import sys
sys.path.append("zhang-shasha/zss")
from simple_tree import *
from compare import *
import numpy as np
import matplotlib.pyplot as plt
from MergeTree import *

def convertToZSSTreeRec(N, NZSS):
    for C in N.children:
        CZSS = Node(C)
        NZSS.addkid(CZSS)
        convertToZSSTreeRec(C, CZSS)

def convertToZSSTree(T):
    rootZSS = Node(T.root)
    convertToZSSTreeRec(T.root, rootZSS)
    return rootZSS

def insertRemoveCost(ZSSNode):
    N = ZSSNode.label
    if N.subdivided:
        return 0
    else:
        P = getParentNotSubdivided(N)
        if not P: #N must be the root
            return 1e9 #Return a very large number
        return np.abs(P.getfVal() - N.getfVal())

def updateCost(ZSSA, ZSSB):
    A = ZSSA.label
    B = ZSSB.label
    cost = np.abs(A.getfVal() - B.getfVal())
    # if A.subdivided or B.subdivided:
    #     print "Matching subdivided, cost = ", cost
    # else:
    #     print "Not matching subdivided, cost = ", cost
    return cost

def doZSSMap(TA, TB):
    subdivideTreesMutual(TA, TB)
    # TA.render(np.array([0, 0]))
    # plt.hold(True)
    # TB.render(np.array([6, 0]))
    # plt.show()

    TA.sortChildrenTotalOrder()
    TB.sortChildrenTotalOrder()

    TAZSS = convertToZSSTree(TA)
    TBZSS = convertToZSSTree(TB)
    return distance(TAZSS, TBZSS, Node.get_children, insertRemoveCost, insertRemoveCost, updateCost)
