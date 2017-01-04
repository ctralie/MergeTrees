"""Wrap around the ZSS library for Zhang/Shasha"""
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
            return np.inf #Roots must match to roots; no deleting allowed
        return np.abs(P.getfVal() - N.getfVal())

def updateCost(ZSSA, ZSSB):
    A = ZSSA.label
    B = ZSSB.label
    cost = np.abs(A.getfVal() - B.getfVal())
    return cost

def doZSSMap(TA, TB):
    subdivideTreesMutual(TA, TB)
    TA.sortChildrenTotalOrder()
    TB.sortChildrenTotalOrder()

    TAZSS = convertToZSSTree(TA)
    TBZSS = convertToZSSTree(TB)

    #Call the ZSS library and my backtracing library
    (KeyrootMaps, KeyrootPtrs, treedists) = distance(TAZSS, TBZSS, Node.get_children, insertRemoveCost, insertRemoveCost, updateCost)
    (Map, BsNotHit) = zssBacktrace(KeyrootPtrs)

    #Now copy over the returned map
    c = ChiralMap(TA, TB)
    c.TA = TA
    c.TB = TB
    c.cost = treedists[-1, -1]
    c.mapsChecked = 'ZSS'
    c.BsNotHit = [N.label for N in BsNotHit]
    for AZSS in Map:
        A = AZSS.label
        B = Map[AZSS]
        if B:
            B = B.label
        c.Map[A] = B
    return c
