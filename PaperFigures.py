import numpy as np
import matplotlib.pyplot as plt
import sys
from MergeTree import *
from TrendingExamples import *
from Laplacian import *
from DGMTools import *

def PDExample():
    t1 = np.linspace(0, 1, 100)
    t1 = np.sqrt(t1)*2*np.pi
    t2 = np.linspace(0, 1, 60)*np.pi/2
    t = np.concatenate((t1, t2))
    #t = np.linspace(0, 2.5*np.pi, 150)
    x = np.cos(t)
    (MT, PS, PD) = mergeTreeFrom1DTimeSeries(x)
    anchorsIdx = [idx for idx in MT]
    (MT2, PS, PD2) = mergeTreeFrom1DTimeSeries(-x)
    anchorsIdx = anchorsIdx + [idx for idx in MT2] + [0, len(x)-1]
    newAnchors = []
    for a in anchorsIdx:
        for k in range(-3, 3):
            if k == 0:
                continue
            idx = a + k
            if idx >= 0 and idx < len(x):
                newAnchors.append(idx)
    anchorsIdx += newAnchors
    anchorsIdx += [115, 120, 125]
    anchorsIdx = np.array(anchorsIdx)
    anchors = x[anchorsIdx]
    anchors[-3::] = [0.5, 0.3, 0.5]
    anchorWeights = np.ones(len(anchors))

    
    ##Original Signal
    x1 = doLaplacianWarp(x, anchorsIdx, anchors, anchorWeights, False)
    
    
    ##Warped Version of x1
    idx = np.arange(0, 100, 4)
    idx = np.concatenate((idx, 100+np.arange(40), 140+np.arange(0, 20, 2)))
    x2 = x1[idx]
    
    ##Switching order of bump
    (X, MT1, PD) = setupTimeSeries(x1)
    #plt.scatter(anchorsIdx, anchors)
    
    #Figure out max index to preserve
    (MT1Dict, PS, PD1) = mergeTreeFrom1DTimeSeries(x1)
    idx = np.max([idx for idx in MT1Dict])
    anchorsIdx[-3::] = [70, 75, 80]
    anchorsIdx = np.array(anchorsIdx.tolist() + [63])
    anchors = np.array(anchors.tolist() + [x1[idx]])
    anchorWeights = np.ones(len(anchors))
    x3 = doLaplacianWarp(x, anchorsIdx, anchors, anchorWeights, False)
    #plt.scatter(anchorsIdx, anchors)
    
    
    ##Reversing signal
    x4 = x1[::-1]
    
    
    
    ###Setup first plot showing conceptually the merge tree
    plt.plot(np.arange(len(x1)), x1, 'b')
    plt.hold(True)
    
    #MT.render(np.array([0, 0]))
    ax = plt.gca()
    #Put x vals at the critical points
    (MT1Dict, PS, PD) = mergeTreeFrom1DTimeSeries(x1)
    idx = [i for i in MT1Dict]
    (MT1Dict, PS, PD) = mergeTreeFrom1DTimeSeries(-x1)
    idx += [i for i in MT1Dict] + [0, len(x1)-1]
    print idx
    ax.set_xticks(idx)
    #Put y ticks at every unique y value
    yvals = MT1.getfValsSorted().tolist()
    ax.set_yticks(yvals)
    plt.grid()
    plt.xlim([-20, len(x)+20])
    #plt.savefig("MergeTreeConceptual.svg")
    
    
    
    ###Setup second plot with functions and merge trees
    yvals = np.sort(yvals)
    ystrs = ['A', 'B', 'C', 'D', 'E']
    [x3, x4] = [x4, x3] #I like this order of presenting them better
    lw1 = 2
    plt.clf()
    plt.figure(figsize=(18, 5))
    plt.subplot2grid((1, 3), (0, 0), colspan = 2)
    ax = plt.gca()
    ax.set_yticks(yvals)
    ax.set_yticklabels(ystrs, fontsize=16)
    ax.set_xticks([])
    plt.grid()
    (X, MT1, PD1) = setupTimeSeries(x1)
    MT1.render(np.array([0, 0]))
    plt.hold(True)
    plt.plot(np.arange(len(x1)), x1, linewidth=lw1)
    
    offx = len(x1) + 30
#    (X, MT2, PD2) = setupTimeSeries(x2)
#    MT2.render(np.array([offx, 0]))
#    plt.plot(np.arange(len(x2)) + offx, x2, linewidth=lw1)
#    
#    offx = offx + len(x2) + 30
#    (X, MT3, PD3) = setupTimeSeries(x3)
#    MT3.render(np.array([offx, 0]))
#    plt.plot(np.arange(len(x3)) + offx, x3, linewidth=lw1)
#    
#    offx = offx + len(x3) + 30
    (X, MT4, PD4) = setupTimeSeries(x4)
    MT4.render(np.array([offx, 0]))
    plt.plot(np.arange(len(x4)) + offx, x4, linewidth=lw1)
    
    plt.xlim([-10, offx + len(x4) + 10])
    plt.ylim([np.min(yvals)-0.2, np.max(yvals)+0.2])
    plt.title("Merge Trees of Signal Snippets")
    #plt.savefig("MergeTreesSnippets.svg", bbox_inches = 'tight')
    
#    plt.clf()
#    plt.figure(figsize=(8, 6))
#    plt.plot(PD1[:, 0], PD1[:, 1], '.')
#    plt.hold(True)
#    plt.plot(PD2[:, 0], PD2[:, 1], '.')
#    plt.plot(PD3[:, 0], PD3[:, 1], '.')
#    plt.plot(PD4[:, 0], PD4[:, 1], '.')
    d = {}
    for i in range(len(yvals)):
        d[yvals[i]] = ystrs[i]
    plt.subplot(133)
    ax = plt.gca()
    ax.set_xticks(PD1[:, 0])
    ax.set_xticklabels([d[i] for i in PD1[:, 0]], fontsize = 16)
    ax.set_yticks(PD1[:, 1])
    ax.set_yticklabels([d[i] for i in PD1[:, 1]], fontsize = 16)
    plotDGM(PD1, sz=40)
    plt.ylim([np.min(yvals)-0.2, np.max(yvals)+0.2])
    plt.xlim([np.min(yvals)-0.2, np.max(yvals)+0.2])
    plt.grid()
    #plt.axis('equal')
    plt.title("Persistence Diagram")
    #plt.savefig("PDSnippets.svg", bbox_inches = 'tight')
    plt.savefig("MergeTreeSnippets.svg", bbox_inches = 'tight')
    

def editExample():
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([0, 10]))
    G = T.root; G.name = "G"
    A = MergeNode(np.array([-5, 0])); A.name = "A"
    F = MergeNode(np.array([2, 8.5])); F.name = "F"
    G.addChildren([A, F])
    D = MergeNode(np.array([0.5, 6])); D.name = "D"
    E = MergeNode(np.array([3.5, 6])); E.name = "E"
    F.addChildren([D, E])
    C = MergeNode(np.array([3, 5])); C.name = "C"
    B = MergeNode(np.array([4, 4])); B.name = "B"
    E.addChildren([C, B])
    T.render(np.array([0, 0]))
    T.updateNodesList()
    
    names = {A:'A', B:'B', C:'C', D:'D', E:'E', F:'F', G:'G'}
    yvals = []
    for node in T.nodesList:
        plt.text(node.X[0]+0.2, node.X[1]+0.2, node.name)
        yvals.append(node.X[1])
    yvals = np.array(yvals)
    yvals = np.unique(yvals)
    ax = plt.gca()
    ax.set_yticks(yvals)
    ax.set_xticks([])
    plt.grid()
    
    plt.show()

if __name__ == '__main__':
    #PDExample()
    editExample()
