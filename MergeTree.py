import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left



##########################################################
#              Partial Order Functions                   #
##########################################################

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


##########################################################
#           Merge Tree Utility Functions                 #
##########################################################

def isAncestor(node, ancestor):
    if not node.parent:
        return False
    if node.parent == ancestor:
        return True
    return isAncestor(node.parent, ancestor)

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

def drawMap(ChiralMap, offsetA, offsetB, yres = 0.5):
    (TA, TB) = (ChiralMap.TA, ChiralMap.TB)
    plt.clf()
    plt.hold(True)
    #First draw the two trees
    ax = plt.subplot(111)
    TA.render(offsetA)
    TB.render(offsetB)
    #Put y ticks at every unique y value
    yvals = TA.getfValsSorted().tolist() + TB.getfValsSorted().tolist()
    yvals = np.sort(np.unique(np.array(yvals)))
    ax.set_yticks(yvals)
    plt.grid()
    plt.title("Cost = %g"%ChiralMap.cost)

    #Now draw arcs between matched nodes and draw Xs over
    #nodes that didn't get matched
    Map = ChiralMap.Map
    for A in Map:
        B = Map[A]
        ax = A.X + offsetA
        if not B:
            #Draw an X over this node
            plt.scatter([ax[0]], [ax[1]], 300, 'k', 'x')
        else:
            bx = B.X + offsetB# + 0.1*np.random.randn(2)
            plt.plot([ax[0], bx[0]], [ax[1], bx[1]], 'b')
    for B in ChiralMap.BsNotHit:
        bx = B.X + offsetB
        plt.scatter([bx[0]], [bx[1]], 300, 'k', 'x')

class DebugOffsets(object):
    def __init__(self, offsetA, offsetB):
        self.offsetA = offsetA
        self.offsetB = offsetB

#TA: Tree A, TB: Tree B,
#Map: Tree map in the form [[NodeA, NodeB], [NodeA, NodeB], ...]
#ia: Index of currently considered node in TA
#BMapped: Boolean array indicating which nodes in B have been hit
#C: Best chiral map found so far
#cost: Current cost
def doBruteForceMapRec(TA, TB, Map, ia, BMapped, C, cost, debug = None):
    #Has every node in A been processed?
    if ia >= len(TA.nodesList):
        C.mapsChecked += 1
        #Add the cost of every untouched node in TB
        BsNotHit = []
        for ib in range(len(BMapped)):
            if not BMapped[ib]:
                NB = TB.nodesList[ib]
                BsNotHit.append(NB)
                cost += np.abs(NB.getfVal() - NB.parent.getfVal())

        if debug:
            #Plot for debugging
            thisC = ChiralMap(TA, TB)
            thisC.cost = cost
            thisC.Map = Map
            thisC.BsNotHit = BsNotHit
            drawMap(thisC, debug.offsetA, debug.offsetB)
            plt.savefig("%i.svg"%C.mapsChecked, bbox_inches = 'tight')

        if cost < C.cost:
            #This is the best found map so far, so update the cost
            #and copy it over
            print "Best map so far found cost %g"%cost
            C.cost = cost
            C.BsNotHit = BsNotHit
            for A in Map:
                C.Map[A] = Map[A]
        return

    #If there are still nodes in A to process, keep building the map
    NA = TA.nodesList[ia]
    fNA = NA.getfVal()
    #Step 1: Come up with all valid nodes that TA[i] can map to
    #in TB including the null node, as well as their costs
    ValidBNodes = []
    for ib in range(len(BMapped)):
        if BMapped[ib]:
            #This node is already in the image of the currently
            #considered map
            continue
        NB = TB.nodesList[ib]
        costAdd = np.abs(fNA - NB.getfVal())
        #Before checking anything else, see if the cost would exceed
        #the best cost found so far
        if cost + costAdd > C.cost:
            continue
        #First, check that ancestral relationships are preserved
        #That is, if A is an ancestor of NA, then
        #Map[A] should be an ancestor of NB, and vice versa
        valid = True
        for A in Map:
            if not Map[A]:
                continue
            #First, check that partial order is preserved
            if TA.orderFn(A, NA) != TB.orderFn(Map[A], NB):
                valid = False
                break
            #Second, check the ancestral condition
            if isAncestor(A, NA) and (not isAncestor(Map[A], NB)):
                valid = False
                break
            if isAncestor(NA, A) and (not isAncestor(NB, Map[A])):
                valid = False
                break
        if valid:
            ValidBNodes.append((ib, NB, costAdd))

    #Step 2: Try each valid target B node in turn, and recurse
    for (ib, NB, costAdd) in ValidBNodes:
        BMapped[ib] = True
        Map[NA] = NB
        doBruteForceMapRec(TA, TB, Map, ia+1, BMapped, C, cost + costAdd, debug)
        BMapped[ib] = False

    #Also include a map to the "null node.""  In this case, the cost is
    #the distance of the unmapped node to its direct parent
    Map[NA] = None
    costAdd = np.abs(fNA - NA.parent.getfVal())
    doBruteForceMapRec(TA, TB, Map, ia+1, BMapped, C, cost + costAdd, debug)
    return C

#TA: First tree, TB: Second Tree
#offsetA, offsetB: For rendering debugging
def doBruteForceMap(TA, TB, debug = None):
    #Step 1: Subdivide TA and TB
    valsA = TA.getfValsSorted()
    valsB = TB.getfValsSorted()
    #Subdivide both edges to make sure internal nodes get matched to internal nodes by horizontal lines
    vals = np.array(valsA.tolist() + valsB.tolist())
    vals = np.sort(np.unique(vals))
    TB.subdivideFromValues(vals)
    TA.subdivideFromValues(vals)
    TA.updateNodesList()
    TB.updateNodesList()

    #Step 2: Map root to root and start recursion
    Map = {TA.root:TB.root}
    C = ChiralMap(TA, TB)
    BMapped = [False]*len(TB.nodesList)
    BMapped[0] = True
    cost = np.abs(TA.root.getfVal() - TB.root.getfVal())
    doBruteForceMapRec(TA, TB, Map, 1, BMapped, C, cost, debug)
    return C

##########################################################
#               Core Merge Tree Objects                  #
##########################################################

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
        self.nodesList = []

    def addOffsetRec(self, node, offset):
        node.X += offset
        for C in node.children:
            self.addOffsetRec(C, offset)

    def addOffset(self, offset):
        self.addOffsetRec(self.root, offset)

    def renderRec(self, node, offset, pointSize = 200):
        X = node.X + offset
        if node.newNode:
            #Render new nodes blue
            plt.scatter(X[0], X[1], pointSize, 'b')
        else:
            plt.scatter(X[0], X[1], pointSize, 'r')
        if node.parent:
            Y = node.parent.X + offset
            plt.plot([X[0], Y[0]], [X[1], Y[1]], 'k')
        for C in node.children:
            self.renderRec(C, offset)

    def render(self, offset, pointSize = 200):
        plt.hold(True)
        self.renderRec(self.root, offset, pointSize)

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

    def updateNodesListRec(self, N):
        self.nodesList.append(N)
        for C in N.children:
            self.updateNodesListRec(C)

    def updateNodesList(self):
        self.nodesList = []
        self.updateNodesListRec(self.root)

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

    #Note: For simplicity of implementation, this
    #function assumes that parent nodes have higher
    #function values than their children
    def subdivideFromValues(self, vals):
        hi = bisect_left(vals, self.root.getfVal())
        self.subdivideEdgesRec(self.root, vals, hi)
