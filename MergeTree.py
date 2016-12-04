import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left



#For trees embedded in the plane corresponding to
#1D functions, make a total order based on the X
#coordinate
def TotalOrder2DX(N1, N2):
    if N1.X[0] <= N2.X[0]:
        return -1
    return 1

#For trees build on functions over R2 (such as SSMs)
#which live in 3D, use the domain to create a partial
#order
def PartialOrder3DXY(N1, N2):
    [X1, Y1] = [N1.X[0], N1.X[1]]
    [X2, Y2] = [N2.X[0], N2.X[1]]
    if X1 <= X2 and Y1 <= Y2:
        return -1
    elif X1 >= X2 and Y1 >= Y2:
        return 1
    return 0

class MergeNode(object):
    #X: Rendering position.  Last coordinate is assumed to be the function value
    #to be the function value
    #newNode: Whether this node was added in a subdivision (for debugging)
    def __init__(self, X, newNode = False):
        self.parent = None
        self.children = []
        self.X = np.array(X, dtype=np.float64)
        self.newNode = newNode

    def getfVal(self):
        return self.X[-1]

    def addChild(self, N):
        self.children.append(N)
        N.parent = self

    def addChildren(self, arr):
        for C in arr:
            self.addChild(C)

#Holds nodes starting at root, and a table of partial order info (-1 for less than, 1 for greater than, 0 for undefined)
class MergeTree(object):
    def __init__(self, orderFn):
        self.root = None
        self.orderFn = orderFn
        self.fVals = []

    def addOffsetRec(self, node, offset):
        node.X += offset
        for C in node.children:
            self.addOffsetRec(C, offset)

    def addOffset(self, offset):
        self.addOffsetRec(self.root, offset)

    def renderRec(self, node, offset):
        X = node.X + offset
        if node.newNode:
            #Render new nodes blue
            plt.scatter(X[0], X[1], 40, 'b')
        else:
            plt.scatter(X[0], X[1], 40, 'r')
        if node.parent:
            Y = node.parent.X + offset
            plt.plot([X[0], Y[0]], [X[1], Y[1]], 'k')
        for C in node.children:
            self.renderRec(C, offset)

    def render(self, offset):
        plt.hold(True)
        self.renderRec(self.root, offset)

    def getfValsSortedRec(self, node):
        self.fVals.append(node.getfVal())
        for n in node.children:
            self.getfValsSortedRec(n)

    #Get a sorted list of all of the function values
    def getfValsSorted(self):
        self.fVals = []
        self.getfValsSortedRec(self.root)
        self.fVals = sorted(self.fVals)
        self.fVals = np.unique(self.fVals)
        return self.fVals

    #hi: Index such that vals[0:hi] < N1.fVal
    def subdivideEdgesRec(self, N1, vals, hi):
        b = N1.getfVal()
        if b <= vals[0]:
            return
        for i in range(len(N1.children)):
            N2 = N1.children[i]
            a = N2.getfVal()
            #Figure out the elements in vals which
            #are in the open interval (a, b)
            lo = bisect_left(vals, a)
            splitVals = []
            for k in range(lo, hi+1):
                if vals[k] <= a or vals[k] >= b:
                    continue
                splitVals.append(vals[k])
            if len(splitVals) > 0:
                #Now split the edge between N1 and N2
                newNodes = []
                for k in range(len(splitVals)):
                    t = (splitVals[k] - a)/float(b - a)
                    X = t*N1.X + (1-t)*N2.X
                    N = MergeNode(X, True)
                    if k > 0:
                        newNodes[k-1].addChild(N)
                    newNodes.append(N)
                #The last node is connected to N2
                newNodes[-1].addChild(N2)
                #Replace N2 in N1's children list with
                #the first node in newNodes
                N1.children[i] = newNodes[0]
                newNodes[0].parent = N1
            self.subdivideEdgesRec(N2, vals, lo)


    #Note: For simplicity of implementation, this
    #function assumes that parent nodes have higher
    #function values than their children
    def subdivideFromValues(self, vals):
        hi = bisect_left(vals, self.root.getfVal())
        self.subdivideEdgesRec(self.root, vals, hi)
