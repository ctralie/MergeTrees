from MergeTree import *
from ZSSMap import *
from DGMTools import *
import time
import matplotlib.pyplot as plt
import scipy.io as sio

def getTrendingLinear(t, a, b):
    x = a*np.cos(t) + b*t
    return x

def getTrendingGaussian(t, a, sigma, b):
    x = a*np.cos(t)*np.exp(-t**2/(2*sigma**2)) + b
    return x

def setupTimeSeries(x):
    N = len(x)
    X = np.zeros((N, 2))
    X[:, 0] = np.arange(N)
    X[:, 1] = x
    (MT, PD) = mergeTreeFrom1DTimeSeries(x)
    MT = wrapMergeTreeTimeSeries(MT, X)
    return (X, MT, PD)

def do4x4Tests(doPlot = False, drawSubdivided = False):
    N = 100
    NPeriods = 5.7
    t1 = np.linspace(0, 1, N)
    t2 = t1**2
    t3 = np.sqrt(t1)
    t4 = t1**0.5 + t1**3
    t4 = t4/np.max(t4)

    ts = [NPeriods*2*np.pi*t for t in [t1, t2, t3, t4]]
    TLUs = [] #Trending Linear Up
    TLDs = [] #Trending Linear Down
    TGUs = [] #Trending Gaussian Up
    TGDs = [] #Trending Gaussian Down
    i = 0
    for t in ts:
        x = getTrendingLinear(t, 1, 0.5*NPeriods/np.max(t))
        sio.savemat("TLU%i.mat"%i, {"x":x})
        TLUs.append(setupTimeSeries(x))
        TLDs.append(setupTimeSeries(x[::-1]))
        c = np.max(x)/2
        y = getTrendingGaussian(t, c, np.max(t)/2, c)
        sio.savemat("TGD%i.mat"%i, {"x":y})
        TGUs.append(setupTimeSeries(y))
        TGDs.append(setupTimeSeries(y[::-1]))
        i += 1

    i = 0
    AllTs = TLUs + TLDs + TGUs + TGDs
    N = len(AllTs)
    #Plot the time series
    if doPlot:
        for i in range(N):
            (X, T, Dgm) = AllTs[i]
            plt.clf()
            plt.plot(X[:, 0], X[:, 1])
            plt.hold(True)
            T.render(np.array([0, 0]))
            plt.savefig("%i.svg"%i)

    #Compute all pairwise distances (check that it's symmetric)
    DMergeTree = np.zeros((N, N))
    DEuclidean = np.zeros((N, N))
    DWasserstein = np.zeros((N, N))
    doWasserstein = True
    for i in range(N):
        print("%i of %i"%(i+1, N))
        (XA, TA, DgmA) = AllTs[i]
        MTTime = 0.0
        WSTime = 0.0
        for j in range(N):
            if i == j:
                continue
            (XB, TB, DgmB) = AllTs[j]
            #Clone so subdivided nodes don't accumulate
            TAClone = TA.clone()
            TBClone = TB.clone()
            DEuclidean[i, j] = np.sqrt(np.sum((XA[:, 1]-XB[:, 1])**2))
            tic = time.time()
            C = getZSSMap(TAClone, TBClone, doPlot)
            toc = time.time()
            MTTime += toc - tic
            if doPlot:
                DMergeTree[i, j] = C.cost
                offset = np.max(XB[:, 0]) - np.min(XA[:, 0]) + 5
                plt.clf()
                drawMap(C, np.array([0, 0]), np.array([offset, 0]), drawSubdivided = False)
                plt.savefig("Map%i_%i.svg"%(i, j))
            else:
                DMergeTree[i, j] = C
            if doWasserstein:
                tic = time.time()
                DWasserstein[i, j] = getWassersteinDist(DgmA, DgmB)
                toc = time.time()
                WSTime += toc-tic
        print "Elapsed Time Merge Tree: ", MTTime
        print "Elapsed Time Wasserstein: ", WSTime
        sio.savemat("PairwiseDs.mat", {"DEuclidean":DEuclidean, "DMergeTree":DMergeTree, "DWasserstein":DWasserstein})

def doStabilityTest():
    mag = 0.1
    N = 100
    t = np.linspace(0, 4*np.pi, N) - np.pi/2
    y = np.sin(t)
    ss = 2*np.pi - np.pi/2
    sc = 3*np.pi - np.pi/2
    n = len(t[t>ss])

    y1 = np.array(y)
    y1[t>ss] = mag*np.sin(np.linspace(0, np.pi, n)) + y[t>ss]
    (X1, T1, Dgm1) = setupTimeSeries(y1)

    y2 = np.array(y)
    y2[t>ss] = -mag*np.sin(np.linspace(0, np.pi, n)) + y[t>ss]
    (X2, T2, Dgm2) = setupTimeSeries(y2)

    plt.subplot(121)
    plt.plot(y1)
    plt.hold(True)
    T1.render(np.array([0, 0]))
    plt.ylim([-1-2*mag, 1+3*mag])
    plt.subplot(122)
    plt.plot(y2)
    plt.hold(True)
    T2.render(np.array([0, 0]))
    plt.ylim([-1-2*mag, 1+3*mag])
    plt.show()

    C = getZSSMap(T1, T2, True)
    offset = np.max(X2[:, 0]) - np.min(X2[:, 0])
    offset *= 1.5
    plt.clf()
    drawMap(C, np.array([0, 0]), np.array([offset, 0]), drawSubdivided = True)
    plt.show()

def testWasserstein():
    N = 100
    NPeriods = 5.7
    t = NPeriods*2*np.pi*np.linspace(0, 1, N)
    x = getTrendingLinear(t, 1, 0.5*NPeriods/np.max(t))
    y = x[::-1] + 0.2
    (X, TX, DgmX) = setupTimeSeries(x)
    (Y, TY, DgmY) = setupTimeSeries(y)
    dist = getWassersteinDist(DgmX, DgmY)
    plot2DGMs(DgmX, DgmY, 'Trending Up', 'Trending Down')
    plt.title("Wasserstein Distance = %g"%dist)
    plt.show()

if __name__ == '__main__':
    do4x4Tests()
    #testWasserstein()
    #doStabilityTest()

if __name__ == '__main__2':
    print("<table>")
    c = 0
    for i in range(4):
        print("<tr>")
        for j in range(4):
            print("<td><img src = \"MergeTreesSVG/%i.svg\"></td>"%c)
            c += 1
        print("</tr>")
    print("</table>")
