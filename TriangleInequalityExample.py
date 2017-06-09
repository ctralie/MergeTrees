import numpy as np
import matplotlib.pyplot as plt
from MergeTree import *
from TrendingExamples import *

def getTrendingSinusoid(NPeriods, N):
    t = np.linspace(np.pi/2, NPeriods*2*np.pi-np.pi/2, N)
    t2 = (t-np.mean(t))**2
    t2 = t2/np.max(t2)
    t2 = 4 + t2
    t2 = t2/np.max(t2)
    y = np.cos(t)*t2
    y = y - np.min(y)
    return y

def doExample():
    lw1 = 4
    x1 = getTrendingSinusoid(3, 100)
    x2 = getTrendingSinusoid(5, 200)
    x3 = np.array(x2)*4
    
    
    print "Making MT1"
    (X1, MT1, PD1) = setupTimeSeries(x1)
    print "Making MT2"
    (X2, MT2, PD2) = setupTimeSeries(x2)
    print "Making MT3"
    (X3, MT3, PD3) = setupTimeSeries(x3)
    print "Finished"
    
    plt.figure(figsize=(18, 6))
    MT1.render(np.array([0, 0]))
    plt.hold(True)
    plt.plot(np.arange(len(x1)), x1, linewidth=lw1)
    
    offx = len(x1) + 30
    MT2.render(np.array([offx, 0]))
    plt.plot(np.arange(len(x2)) + offx, x2, linewidth=lw1)
    
    offx = offx + len(x2) + 30
    MT3.render(np.array([offx, 0]))
    plt.plot(np.arange(len(x3)) + offx, x3, linewidth=lw1)
    

    plt.savefig("Trees.svg", bbox_inches='tight')
    
    plt.clf()
    
    #Now do maps
    TA = MT1.clone()
    TB = MT2.clone()
    C = getZSSMap(TA, TB, True)
    drawMap(C, np.array([0, 0]), np.array([len(x1)*1.5, 0]), drawSubdivided = False, drawCurved = False)
    plt.savefig("MapBlueGreen.svg")
    
    plt.clf()
    TA = MT1.clone()
    TB = MT3.clone()
    C = getZSSMap(TA, TB, True)
    drawMap(C, np.array([0, 0]), np.array([len(x1)*1.5, 0]), drawSubdivided = False, drawCurved = False)
    plt.savefig("MapBlueRed.svg")
    
    plt.clf()
    TA = MT2.clone()
    TB = MT3.clone()
    C = getZSSMap(TA, TB, True)
    drawMap(C, np.array([0, 0]), np.array([len(x2)*1.5, 0]), drawSubdivided = False, drawCurved = False)
    plt.savefig("MapGreenRed.svg")

if __name__ == '__main__':
    doExample()
