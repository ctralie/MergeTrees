"""
Wrap around the bottleneck distance executable from Dionysus, and provide
some utility functions for plotting
"""
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os

def plotDGM(dgm, color = 'b', sz = 20, label = 'dgm'):
    if dgm.size == 0:
        return
    # Create Lists
    # set axis values
    axMin = np.min(dgm)
    axMax = np.max(dgm)
    axRange = axMax-axMin;
    # plot points
    plt.scatter(dgm[:, 0], dgm[:, 1], sz, color,label=label)
    plt.hold(True)
    # plot line
    plt.plot([axMin-axRange/5,axMax+axRange/5], [axMin-axRange/5, axMax+axRange/5],'k');
    # adjust axis
    #plt.axis([axMin-axRange/5,axMax+axRange/5, axMin-axRange/5, axMax+axRange/5])
    # add labels
    plt.xlabel('Time of Birth')
    plt.ylabel('Time of Death')

def plot2DGMs(P1, P2, l1 = 'Diagram 1', l2 = 'Diagram 2'):
    plotDGM(P1, 'r', 10, label = l1)
    plt.hold(True)
    plt.plot(P2[:, 0], P2[:, 1], 'bx', label = l2)
    plt.legend()
    plt.xlabel("Birth Time")
    plt.ylabel("Death Time")

def savePD(filename, I):
    if os.path.exists(filename):
        os.remove(filename)
    fout = open(filename, "w")
    for i in range(I.shape[0]):
        fout.write("%g %g"%(I[i, 0], I[i, 1]))
        if i < I.shape[0]-1:
            fout.write("\n")
    fout.close()

def getBottleneckDist(PD1, PD2):
    savePD("PD1.txt", PD1)
    savePD("PD2.txt", PD2)
    proc = subprocess.Popen(["./bottleneck", "PD1.txt", "PD2.txt"], stdout=subprocess.PIPE)
    return float(proc.stdout.readline())
