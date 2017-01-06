from MergeTree import *
from ZSSMap import *
import timeseries
import time
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    N = 100
    NPeriods = 5.5
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
        x = getTrendingLinear(t, 1, NPeriods/np.max(t))
        (s, X) = makeTimeSeriesGDA(x)
        TLUs.append(wrapGDAMergeTreeTimeSeries(s, X))
        (s, X) = makeTimeSeriesGDA(x[::-1])
        TLDs.append(wrapGDAMergeTreeTimeSeries(s, X))
        c = np.max(x)/2
        y = getTrendingGaussian(t, c, np.max(t)/2, c)
        (s, X) = makeTimeSeriesGDA(y)
        TGDs.append(wrapGDAMergeTreeTimeSeries(s, X))
        (s, X) = makeTimeSeriesGDA(y[::-1])
        TGUs.append(wrapGDAMergeTreeTimeSeries(s, X))
