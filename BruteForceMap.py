from MergeTree import *

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
                if not NB.subdivided: #Cost of deleting subdivided nodes is 0
                    P = getParentNotSubdivided(NB)
                    cost += np.abs(NB.getfVal() - P.getfVal())

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
        if NA.subdivided and NB.subdivided:
            #No need to map subdivided nodes to subdivided nodes
            continue
        BMapped[ib] = True
        Map[NA] = NB
        doBruteForceMapRec(TA, TB, Map, ia+1, BMapped, C, cost + costAdd, debug)
        BMapped[ib] = False

    #Also include a map to the "null node.""  In this case, the cost is
    #the distance of the unmapped node to its direct parent
    #(not including subdivided nodes)
    Map[NA] = None
    costAdd = 0
    if not NA.subdivided:
        P = getParentNotSubdivided(NA)
        costAdd = np.abs(fNA - P.getfVal())
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
