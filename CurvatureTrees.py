import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import glob
from CurvatureTools import *
from BottleneckWrapper import *
from TrendingExamples import *
from CoverTree import *
import subprocess
import _SequenceAlignment as SAC

def drawLineColored(idx, x, C):
    plt.hold(True)
    for i in range(len(x)-1):
        plt.plot(idx[i:i+2], x[i:i+2], c=C[i, :])

def plotCurvature(x, v, c, C, Colors, MT, DGM, plotArrows = False):
    plt.subplot(131)
    plt.scatter(x[:, 0], x[:, 1], 20, c=Colors, cmap = 'Spectral', edgecolor = 'none')
    plt.title("Parameterized Curve")
    #Plot velocity points
    plt.hold(True)
    ax = plt.gca()
    if plotArrows:
        for i in range(1, x.shape[0], 20):
            v1 = x[i, :]
            #Draw velocity arrow
            v2 = v1 + 10*v[i, :]
            diff = v2-v1
            ax.arrow(v1[0], v1[1], diff[0], diff[1], head_width = 2, head_length = 4, fc = 'k', ec = 'k')
            #Draw curvature arrow
            v2 = v1 + 50*c[i, :]
            diff = v2-v1
            ax.arrow(v1[0], v1[1], diff[0], diff[1], head_width = 2, head_length = 4, fc = 'r', ec = 'r')
            #ax.annotate("%i"%i, xy=(v1[0], v1[1]))
    plt.axis('equal')
    plt.subplot(132)
    plt.scatter(np.arange(len(C)), C, 20, c=Colors, edgecolor = 'none')
    plt.hold(True)
    drawLineColored(np.arange(len(C)), C, Colors)
    plt.plot([0, len(C)], [0, 0], 'k', linestyle='--')
    MT.render(np.array([0, 0]), pointSize=60, lineWidth=2)
    plt.ylim([-0.2, 0.5])
    plt.xlim([0, len(C)])
    plt.xlabel("Time Index")
    plt.ylabel("Smoothed Curvature")
    plt.title("Curvature Estimate")
    plt.subplot(133)
    plotDGM(DGM)
    plt.plot([-0.5, 0.5], [-0.5, 0.5], 'k')
    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    plt.title("Persistence Diagram")

def tryTriangleIneqExample(dirName, NClasses, NPerClass, sigma, i, j, k, doPlot = True):
    N = NClasses*NPerClass
    idx = 0
    (Ti, Tj, Tk) = (None, None, None)
    filenames = []
    for cl in range(1, NClasses+1):
        for num in range(1, NPerClass+1):
            filenames.append("%s/Class%i_Sample%i.mat"%(dirName, cl, num))

    MTs = []
    idxs = [i, j, k]
    for idx in idxs:
        print filenames[idx]
        x = sio.loadmat(filenames[idx])['x']
        x = np.array(x, dtype=np.float32)
        curvs = getCurvVectors(x, 2, sigma)
        v = curvs[1]
        vMag = np.sum(v**2, 1)
        vMag[vMag == 0] = 1
        vMag = np.sqrt(vMag)
        v = v/vMag[:, None]
        c = curvs[2]
        Curv = np.cross(v, c, 1)
        MTs.append(setupTimeSeries(Curv))

    #Now compute all of the pairwise merge tree and persistence diagram distances
    D = np.zeros((3, 3))
    for i in range(3):
        (XA, TA, DgmA) = MTs[i]
        print "Doing %i of 3..."%i
        tic = time.time()
        for j in range(i+1, 3):
            (XB, TB, DgmB) = MTs[j]
            #Clone so subdivided nodes don't accumulate
            TAClone = TA.clone()
            TBClone = TB.clone()
            C = getZSSMap(TAClone, TBClone, doPlot)
            if doPlot:
                offset = np.max(XB[:, 0]) - np.min(XA[:, 0]) + 5
                plt.clf()
                drawMap(C, np.array([0, 0]), np.array([offset, 0]), drawSubdivided = False)
                plt.savefig("Map%i_%i.svg"%(idxs[i], idxs[j]))

def getCurvatureMergeTrees(dirName, NClasses, NPerClass, sigma, doPlot = True):
    plt.figure(figsize=(18,5))
    N = NClasses*NPerClass
    i = 0
    MTs = []
    for cl in range(1, NClasses+1):
        for num in range(1, NPerClass+1):
            filename = "%s/Class%i_Sample%i.mat"%(dirName, cl, num)
            print filename
            x = sio.loadmat(filename)['x']
            x = np.array(x, dtype=np.float32)
            curvs = getCurvVectors(x, 2, sigma)
            v = curvs[1]
            vMag = np.sum(v**2, 1)
            vMag[vMag == 0] = 1
            vMag = np.sqrt(vMag)
            v = v/vMag[:, None]
            c = curvs[2]
            Curv = np.cross(v, c, 1)
            MTs.append(setupTimeSeries(Curv))
            if doPlot:
                cmap = plt.get_cmap('Spectral')
                Colors = cmap(np.array(np.round(np.linspace(0, 255, len(Curv))), dtype=np.int32))
                Colors = Colors[:, 0:3]
                plt.clf()
                plotCurvature(x, v, c, Curv, Colors, MTs[-1][1], MTs[-1][2])
                plt.savefig("%s.svg"%filename, bbox_inches='tight')

    #Now compute all of the pairwise merge tree and persistence diagram distances
    DMergeTree = np.zeros((N, N))
    DWasserstein = np.zeros((N, N))
    DBottleneck = np.zeros((N, N))
    DDTW = np.zeros((N, N))
    for i in range(N):
        (XA, TA, DgmA) = MTs[i]
        print "Doing %i of %i..."%(i, N)
        tic = time.time()
        for j in range(N):
            (XB, TB, DgmB) = MTs[j]
            #Clone so subdivided nodes don't accumulate
            TAClone = TA.clone()
            TBClone = TB.clone()
            C = getZSSMap(TAClone, TBClone, doPlot)
            if doPlot:
                DMergeTree[i, j] = C.cost
                offset = np.max(XB[:, 0]) - np.min(XA[:, 0]) + 5
                plt.clf()
                drawMap(C, np.array([0, 0]), np.array([offset, 0]), drawSubdivided = False)
                plt.savefig("Map%i_%i.svg"%(i, j))
            else:
                DMergeTree[i, j] = C
            DWasserstein[i, j] = getWassersteinDist(DgmA, DgmB)
            DBottleneck[i, j] = getBottleneckDist(DgmA, DgmB)

            #Compute DTW
            x1 = XA[:, 1].flatten()
            x2 = XB[:, 1].flatten()
            CSM = np.abs(x1[:, None] - x2[None, :])
            DDTW[i, j] = SAC.DTW(CSM)
        toc = time.time()
        print "Elapsed Time: ", toc-tic
        sio.savemat("DCurvatures.mat", {"DMergeTree":DMergeTree, "DWasserstein":DWasserstein, "DBottleneck":DBottleneck, "DDTW":DDTW})

def plotCurve(x):
    cmap = plt.get_cmap('Spectral')
    Colors = cmap(np.array(np.round(np.linspace(0, 255, x.shape[0])), dtype=np.int32))
    Colors = Colors[:, 0:3]
    plt.scatter(x[:, 0], x[:, 1], 20, c=Colors, cmap = 'Spectral', edgecolor = 'none')

def plotHammerCoverTree():
    #First figure out plotting range over all hammers
    res = 0
    for i in range(1, 21):
        filename = "hmm_gpd/bicego_data/Class7_Sample%i.mat"%i
        x = sio.loadmat(filename)['x']
        x = x - np.minimum(x, 1)[None, :]
        res = max(res, np.max(x))
    print "res = ", res

    from ete3 import Tree, TreeStyle, Tree, ImgFace, TextFace, add_face_to_node
    D = sio.loadmat("DCurvatures.mat")
    DMergeTree = D['DMergeTree']
    DWass = D['DWasserstein']
    DMergeTree = DMergeTree[-20::, -20::]
    DWass = DWass[-20::, -20::]


    TCMT = CoverTree()
    seedidx = np.argmin(np.sum(DMergeTree, 1))
    TCMT.construct(DMergeTree, seedidx)
    TB = CoverTree()
    seedidx = np.argmin(np.sum(DWass, 1))
    TB.construct(DWass, seedidx)

    t = TCMT.getETETree(names = [i+1 for i in range(20)])
    #t = TB.getETETree(names = [i+1 for i in range(20)])
    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.show_branch_length = False
    #ts.mode = "c"
    ts.scale = 200
    plt.figure(figsize=(5, 5))
    def pictureLayout(node):
        #http://etetoolkit.org/docs/latest/tutorial/tutorial_drawing.html
        if not type(node.name) == int:
            return
        filename = "hmm_gpd/bicego_data/Class7_Sample%s.mat"%node.name
        x = sio.loadmat(filename)['x']
        print x.shape
        #x = x - np.minimum(x, 1)[None, :]
        plt.clf()
        plotCurve(x)
        plt.xlim([-20, 150])
        plt.ylim([-20, 150])
        #plt.axis('equal')
        plt.axis('off')
        #plt.xlim([0, res])
        #plt.ylim([0, res])
        imname = "Hammer%s.png"%node.name
        plt.savefig(imname, bbox_inches = 'tight')
        #convert Hammer1.png -background none -transparent white -flatten out.png
        #subprocess.call(["convert", imname, "-background", "none", "-transparent", "white", "-flatten", imname])
        subprocess.call(["mogrify", "-resize", "100x100", imname])
        subprocess.call(["mogrify", "-trim", imname])

        F = ImgFace(imname)
        add_face_to_node(F, node, column=0, position="branch-right")

        F = TextFace(node.name, tight_text = True)
        add_face_to_node(F, node, column=0, position="branch-top")

    def textLayout(node):
        #http://etetoolkit.org/docs/latest/tutorial/tutorial_drawing.html
        if not type(node.name) == int:
            return
        F = TextFace(node.name)
        add_face_to_node(F, node, column=0, position="branch-right")

    ts.layout_fn = pictureLayout
    ts.rotation = 90
    #t.show(tree_style=ts)
    #print t
    t.render("HammerTree.svg", tree_style = ts, w=800)

if __name__ == '__main__':
    #plotHammerCoverTree()
    #getCurvatureMergeTrees('hmm_gpd/bicego_data', 7, 20, 10, doPlot = False)
    tryTriangleIneqExample('hmm_gpd/bicego_data', 7, 20, 10, 12, 99, 104)
