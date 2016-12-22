#Wrap around the ZSS library for Zhang/Shasha
from zss import distance, Node
import numpy as np
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
    return np.abs(A.getfVal() - B.getfVal())

def doZSSMap(TA, TB):
    subdivideTreesMutual(TA, TB)
    TA.sortChildrenTotalOrder()
    TB.sortChildrenTotalOrder()
    TAZSS = convertToZSSTree(TA)
    TBZSS = convertToZSSTree(TB)
    return distance(TAZSS, TBZSS, Node.get_children, insertRemoveCost, insertRemoveCost, updateCost)
