import numpy as np
import matplotlib.pyplot as plt

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
    #fVal: Function Value
    #X: Rendering position (not used in actual merge tree comparisons but used to help partial order)
    def __init__(self, fVal, X):
        self.parent = None
        self.children = []
        self.fVal = fVal #Function value
        self.X = X

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

    def renderRec(self, node, offset):
        X = node.X + offset
        plt.scatter(X[0], X[1], 40, 'r')
        if node.parent:
            Y = node.parent.X + offset
            plt.plot([X[0], Y[0]], [X[1], Y[1]], 'k')
        for C in node.children:
            self.renderRec(C, offset)

    def render(self, offset):
        self.renderRec(self.root, offset)
        plt.grid()

    def subdivideFromOtherTree(self, T):
        print "TODO"
