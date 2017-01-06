from MergeTree import *
from ZSSMap import *
from BottleneckWrapper import *
import timeseries
import time
import matplotlib.pyplot as plt
import scipy.io as sio

def getTrendingLinear(t, a, b):
    x = a*np.cos(t) + b*t
    return x

def getTrendingGaussian(t, a, sigma, b):
    x = a*np.cos(t)*np.exp(-t**2/(2*sigma**2)) + b
    return x

def makeTimeSeriesGDA(x):
    N = len(x)
    X = np.zeros((N, 2))
    X[:, 0] = np.arange(N)
    X[:, 1] = x
    s = timeseries.Signal(x)
    return (s, X)

def do4x4Tests():
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
    for t in ts:
        x = getTrendingLinear(t, 1, 0.5*NPeriods/np.max(t))
        (s, X) = makeTimeSeriesGDA(x)
        TLUs.append((X, wrapGDAMergeTreeTimeSeries(s, X)))
        (s, X) = makeTimeSeriesGDA(x[::-1])
        TLDs.append((X, wrapGDAMergeTreeTimeSeries(s, X)))
        c = np.max(x)/2
        y = getTrendingGaussian(t, c, np.max(t)/2, c)
        (s, X) = makeTimeSeriesGDA(y)
        TGDs.append((X, wrapGDAMergeTreeTimeSeries(s, X)))
        (s, X) = makeTimeSeriesGDA(y[::-1])
        TGUs.append((X, wrapGDAMergeTreeTimeSeries(s, X)))

    i = 0
    doPlot = False
    drawSubdivided = False
    AllTs = TLUs + TLDs + TGUs + TGDs
    N = len(AllTs)
    #Plot the time series
    if doPlot:
        for i in range(N):
            (X, (T, Dgm)) = AllTs[i]
            plt.clf()
            plt.plot(X[:, 0], X[:, 1])
            plt.hold(True)
            T.render(np.array([0, 0]))
            plt.savefig("%i.svg"%i)

    #Compute all pairwise distances (check that it's symmetric)
    DMergeTree = np.zeros((N, N))
    DEuclidean = np.zeros((N, N))
    DBottleneck = np.zeros((N, N))
    doBottleneck = True
    for i in range(N):
        print("%i of %i"%(i+1, N))
        (XA, (TA, DgmA)) = AllTs[i]
        TAClone = TA.clone() #Clone so subdivided nodes don't accumulate
        tic = time.time()
        for j in range(N):
            if i == j:
                continue
            (XB, (TB, DgmB)) = AllTs[j]
            TBClone = TB.clone()
            DEuclidean[i, j] = np.sqrt(np.sum((XA[:, 1]-XB[:, 1])**2))
            C = getZSSMap(TAClone, TBClone, doPlot)
            if doPlot:
                DMergeTree[i, j] = C.cost
                offset = np.max(XB[:, 0]) - np.min(XA[:, 0]) + 5
                plt.clf()
                drawMap(C, np.array([0, 0]), np.array([offset, 0]), drawSubdivided = False)
                plt.savefig("Map%i_%i.svg"%(i, j))
            else:
                DMergeTree[i, j] = C
            if doBottleneck:
                DBottleneck[i, j] = getBottleneckDist(DgmA, DgmB)
        print("Elapsed time: ", time.time() - tic)
        sio.savemat("PairwiseDs.mat", {"DEuclidean":DEuclidean, "DMergeTree":DMergeTree, "DBottleneck":DBottleneck})

def testBottleneck():
    N = 100
    NPeriods = 5.7
    t = NPeriods*2*np.pi*np.linspace(0, 1, N)
    x = getTrendingLinear(t, 1, 0.5*NPeriods/np.max(t))
    y = x[::-1] + 0.2
    (sx, X) = makeTimeSeriesGDA(x)
    (TX, DgmX) = wrapGDAMergeTreeTimeSeries(sx, X)
    (sy, Y) = makeTimeSeriesGDA(y)
    (TY, DgmY) = wrapGDAMergeTreeTimeSeries(sy, Y)
    dist = getBottleneckDist(DgmX, DgmY)
    plot2DGMs(DgmX, DgmY, 'Trending Up', 'Trending Down')
    plt.title("Bottleneck Distance = %g"%dist)
    plt.show()

if __name__ == '__main__':
    do4x4Tests()
    #testBottleneck()

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
