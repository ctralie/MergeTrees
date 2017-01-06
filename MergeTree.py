import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left

##########################################################
#              Partial Order Functions                   #
##########################################################

def TotalOrder2DX(N1, N2):
    """
    For trees embedded in the plane corresponding to
    1D functions, make a total order based on the X
    coordinate
    """
    if N1.X[0] <= N2.X[0]:
        return -1
    return 1

def PartialOrder3DXY(N1, N2):
    """
    For trees build on functions over R2 (such as SSMs)
    #which live in 3D, use the domain to create a partial
    order
    """
    [X1, Y1] = [N1.X[0], N1.X[1]]
    [X2, Y2] = [N2.X[0], N2.X[1]]
    if X1 <= X2 and Y1 <= Y2:
        return -1
    elif X1 >= X2 and Y1 >= Y2:
        return 1
    return 0


##########################################################
#           Merge Tree Utility Functions                 #
##########################################################

def isAncestor(node, ancestor):
    if not node.parent:
        return False
    if node.parent == ancestor:
        return True
    return isAncestor(node.parent, ancestor)


def getParentNotSubdivided(N):
    if not N.parent:
        return None
    if N.parent.subdivided:
        return getParentNotSubdivided(N.parent)
    else:
        return N.parent

def subdivideTreesMutual(TA, TB):
    valsA = TA.getfValsSorted()
    valsB = TB.getfValsSorted()
    #Subdivide both edges to make sure internal nodes get matched to internal nodes by horizontal lines
    vals = np.array(valsA.tolist() + valsB.tolist())
    vals = np.sort(np.unique(vals))
    TB.subdivideFromValues(vals)
    TA.subdivideFromValues(vals)
    TA.updateNodesList()
    TB.updateNodesList()

##########################################################
#                   Merge Tree Maps                      #
##########################################################

class ChiralMap(object):
    def __init__(self, TA, TB):
        self.TA = TA
        self.TB = TB
        self.cost = np.inf
        self.Map = {}
        self.mapsChecked = 0
        self.BsNotHit = []

def drawMap(ChiralMap, offsetA, offsetB, yres = 0.5, drawSubdivided = True):
    (TA, TB) = (ChiralMap.TA, ChiralMap.TB)
    plt.clf()
    plt.hold(True)
    #First draw the two trees
    ax = plt.subplot(111)
    TA.render(offsetA, drawSubdivided = drawSubdivided)
    TB.render(offsetB, drawSubdivided = drawSubdivided)
    #Put y ticks at every unique y value
    yvals = TA.getfValsSorted().tolist() + TB.getfValsSorted().tolist()
    yvals = np.sort(np.unique(np.array(yvals)))
    ax.set_yticks(yvals)
    plt.grid()
    plt.title("Cost = %g\nmapsChecked = %s"%(ChiralMap.cost, ChiralMap.mapsChecked))

    #Now draw arcs between matched nodes and draw Xs over
    #nodes that didn't get matched
    Map = ChiralMap.Map
    for A in Map:
        B = Map[A]
        ax = A.X + offsetA
        if not A.subdivided or drawSubdivided:
            if not B:
                #Draw an X over this node
                plt.scatter([ax[0]], [ax[1]], 300, 'k', 'x')
            else:
                bx = B.X + offsetB# + 0.1*np.random.randn(2)
                plt.plot([ax[0], bx[0]], [ax[1], bx[1]], 'b')
    for B in ChiralMap.BsNotHit:
        if not B.subdivided or drawSubdivided:
            bx = B.X + offsetB
            plt.scatter([bx[0]], [bx[1]], 300, 'k', 'x')

class DebugOffsets(object):
    def __init__(self, offsetA, offsetB):
        self.offsetA = offsetA
        self.offsetB = offsetB


##########################################################
#               Core Merge Tree Objects                  #
##########################################################

class MergeNode(object):
    """
    X: Rendering position.  Last coordinate is assumed to be the function value
    to be the function value
    subdivided: Whether this node was added in a subdivision
    """
    def __init__(self, X, subdivided = False):
        self.parent = None
        self.children = []
        self.X = np.array(X, dtype=np.float64)
        self.subdivided = subdivided

    def getfVal(self):
        return self.X[-1]

    def addChild(self, N):
        self.children.append(N)
        N.parent = self

    def addChildren(self, arr):
        for C in arr:
            self.addChild(C)

    def __str__(self):
        return "Node: X = %s, Subdivided = %i"%(self.X, self.subdivided)

class MergeTree(object):
    """
    Holds nodes starting at root, and a table of partial order info (-1 for less than, 1 for greater than, 0 for undefined)
    """
    def __init__(self, orderFn):
        self.root = None
        self.orderFn = orderFn
        self.fVals = []
        self.nodesList = []

    def addOffsetRec(self, node, offset):
        node.X += offset
        for C in node.children:
            self.addOffsetRec(C, offset)

    def addOffset(self, offset):
        self.addOffsetRec(self.root, offset)

    def renderRec(self, node, offset, drawSubdivided = True, pointSize = 200):
        X = node.X + offset
        if node.subdivided:
            #Render new nodes blue
            if drawSubdivided:
                plt.scatter(X[0], X[1], pointSize, 'b')
        else:
            plt.scatter(X[0], X[1], pointSize, 'r')
        if node.parent:
            Y = node.parent.X + offset
            plt.plot([X[0], Y[0]], [X[1], Y[1]], 'k')
        for C in node.children:
            self.renderRec(C, offset, drawSubdivided, pointSize)

    def render(self, offset, drawSubdivided = True, pointSize = 200):
        plt.hold(True)
        self.renderRec(self.root, offset, drawSubdivided, pointSize)

    def sortChildrenTotalOrderRec(self, N):
        N.children = sorted(N.children, cmp=self.orderFn)
        for C in N.children:
            self.sortChildrenTotalOrderRec(C)


    def sortChildrenTotalOrder(self):
        """
        Sort the children by their total order (behavior undefined
        if orderFn is a partial order)
        """
        self.sortChildrenTotalOrderRec(self.root)

    def getfValsSortedRec(self, node):
        self.fVals.append(node.getfVal())
        for n in node.children:
            self.getfValsSortedRec(n)

    def getfValsSorted(self):
        """Get a sorted list of all of the function values"""
        self.fVals = []
        self.getfValsSortedRec(self.root)
        self.fVals = sorted(self.fVals)
        self.fVals = np.unique(self.fVals)
        return self.fVals

    def updateNodesListRec(self, N):
        self.nodesList.append(N)
        for C in N.children:
            self.updateNodesListRec(C)

    def updateNodesList(self):
        self.nodesList = []
        self.updateNodesListRec(self.root)

    def getCriticalPtsList(self):
        """Return an Nxd numpy array of the N critical points"""
        self.updateNodesList()
        N = len(self.nodesList)
        X = np.zeros((N, self.nodesList[0].X.size))
        for i in range(N):
            X[i, :] = self.nodesList[i].X
        return X

    def subdivideEdgesRec(self, N1, vals, hi):
        """
        Recursive helper function for subdividing edges with all elements
        in "vals"
        hi: Index such that vals[0:hi] < N1.fVal
        """
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
                if k >= len(vals):
                    continue
                if vals[k] <= a or vals[k] >= b:
                    continue
                splitVals.append(vals[k])
            #Nodes were sorted in increasing order of height
            #but need to add them in decreasing order
            #for the parent relationships to work out
            splitVals = splitVals[::-1]
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

    def subdivideFromValues(self, vals):
        """
        Note: For simplicity of implementation, this
        function assumes that parent nodes have higher
        function values than their children
        """
        hi = bisect_left(vals, self.root.getfVal())
        self.subdivideEdgesRec(self.root, vals, hi)

def wrapGDAMergeTreeTimeSeries(s, X):
    """
    s is a time series from the GDA library, X is an Nx2 numpy
    array of the corresponding coordinates
    """
    T = MergeTree(TotalOrder2DX)
    y = X[:, 1]
    MT = s.pers.mergetree
    if len(MT) == 0: #Boundary case
        return T
    nodes = {}
    #Construct all node objects
    root = None
    maxVal = -np.inf
    for idx0 in MT:
        for idx in [idx0] + list(MT[idx0]):
            nodes[idx] = MergeNode(X[idx, :])
            if y[idx] > maxVal:
                root = nodes[idx]
                maxVal = y[idx]
    T.root = root
    #Create all branches
    for idx in MT:
        for cidx in MT[idx]:
            nodes[idx].addChild(nodes[cidx])
    return T
