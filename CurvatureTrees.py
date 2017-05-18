import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import glob
from CurvatureTools import *
from BottleneckWrapper import *
from TrendingExamples import *

def drawLineColored(idx, x, C):
    plt.hold(True)
    for i in range(len(x)-1):
        plt.plot(idx[i:i+2], x[i:i+2], c=C[i, :])

def plotCurvature(x, v, c, C, Colors, MT, DGM, plotArrows = False):
    plt.subplot(131)
    plt.scatter(x[:, 0], x[:, 1], 20, c=Colors, cmap = 'Spectral', edgecolor = 'none')
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
    MT.render(np.array([0, 0]), pointSize=20)
    plt.plot([0, len(C)], [0, 0], 'k', linestyle='--')
    plt.ylim([-0.5, 0.5])
    plt.xlim([0, len(C)])
    plt.subplot(133)
    plotDGM(DGM)
    plt.plot([-0.5, 0.5], [-0.5, 0.5], 'k')
    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])

def getCurvatureMergeTrees(dirName, NClasses, NPerClass, sigma, doPlot = True):
    sigma = 10
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
        toc = time.time()
        print "Elapsed Time: ", toc-tic
        sio.savemat("DCurvatures.mat", {"DMergeTree":DMergeTree, "DWasserstein":DWasserstein, "DBottleneck":DBottleneck})


if __name__ == '__main__':
    getCurvatureMergeTrees('hmm_gpd/bicego_data', 7, 20, 20, doPlot = False)
