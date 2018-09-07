import numpy as np
import _pickle as pickle
import FMC
import os
import itertools as it
import matplotlib.pylab as plt
from scipy.interpolate import griddata


b = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoFiles/MoScans/0DegreeContactCapture.p','rb'))
d = FMC.EstimateProbeDelays(b['AScans'][0], 25., 0.6, 25.4, c=6.4)[0]

yholes = np.array([6.35, 12.7, 19.05, 25.4, 31.75, 38.1, 68.65, 62.3, 55.95, 49.6, 43.25, 36.9])
Yholes = np. array([9.525, 22.225, 34.925, 65.475, 52.775, 40.075])
holesize = np.array([1, 1, 2, 2, 3, 3, 3, 3])

X = np.arange(-21,21,1)
Y = np.arange(1,73,1)

pth = '/mnt/c/Users/mmarvasti/Desktop/MoFiles/MoScans/ContactCalibrationScans/'
Dir = os.listdir(pth)

Amp = []
XX = []
YY = []

for j in range(len(Dir)):

    scan = pickle.load(open(pth + Dir[j],'rb'))

    F = FMC.LinearCapture(25,scan['AScans'],0.6,64,d)

    F.ProcessScans(50,70)

    print('Processing Scan ' + str(j) + ' is Done')

    F.GetContactDelays(X,Y,6.3)

    I = [F.ApplyTFM(i) for i in range(len(F.AScans))]

    print(str(j)+'Done')

    x = np.zeros(len(I))
    y = np.zeros(len(I))
    amp = np.zeros(len(I))

    for i in range(len(I)):

        ind = np.unravel_index(np.argmax(np.abs(I[i])),I[i].shape)

        F.GetContactDelays(np.arange(X[ind[1]]-5.,X[ind[1]]+5.,0.1),np.arange(Y[ind[0]]-5.,Y[ind[0]]+5.,0.1),6.3)

        II = F.ApplyTFM(i)

        iind = np.unravel_index(np.argmax(np.abs(II)), II.shape)

        x[i] = F.xRange[iind[1]]
        y[i] = F.yRange[iind[0]]
        amp[i] = np.abs(II[iind[0],iind[1]])

        print(str(i)+' ' + str(j))

    Amp.append(amp)
    XX.append(x)
    YY.append(y)

pickle.dump({'Amplitude':Amp, 'x':XX, 'y':YY}, open(pth+'CalBlockCoordinatesandAmplitudes.p','wb'))

# xx = []
# yy = []
# amps = []
# ind = []
#
# for i in range(len(Amp)):
#
#     xi = []
#     yi = []
#     amp = []
#     indi = []
#
#     for j in range(len(Amp[i])):
#
#         if (i<6):
#             YH = yholes
#         else:
#             YH = Yholes
#
#         if ((np.any(np.abs(YH - YY[i][j])<2*holesize[i]))&(XX[i][j]>=-19.2)&(XX[i][j]<=19.2)):
#
#             xi.append(XX[i][j])
#             yi.append(XX[i][j])
#             amp.append(Amp[i][j])
#             indi.append(j)
#
#     xx.append(xi)
#     yy.append(yi)
#     amps.append(amp)
#     ind.append(indi)
#
# pickle.dump({'Amplitude':amps, 'x':xx, 'y':yy}, open(pth+'CalBlockCoordinatesandAmplitudesFiltered.p','wb'))

# x = xx[0] + xx[1]
# y = yy[0] + yy[1]
# amps = amps[0] + amps[1]
# G = list(80/array(amps))
#
# dm = 0.1
# m = np.arange(-5,5,dm)
#
# I = []
#
# for i in range(len(Dir)):
#
#     IL = []
#     scan = pickle.load(open(pth + Dir[i],'rb'))
#
#     xp = []
#     yp = []
#
#     for j in range(len(xx[i])):
#
#         phi = np.arctan(xx[i][j]/yy[i][j])
#         xpi = xx[i][j] + m*np.cos(phi)
#         ypi = yy[i][j] - m*np.sin(phi)
#
#         xpii = []
#         ypii = []
#
#         for k in range(len(xpi)):
#             if ((xpi[k]=>-19.2)&(xpi[k]<=19.2)&(ypi[k]>0)&(ypi[k]<75)):
#                 xpii.append(xpi[k])
#                 ypii.append(ypi[k])
#
#         xp.append(xpii)
#         yp.append(ypii)
#
#     # F = FMC.LinearCapture(25,scan['AScans'],0.6,64,d)
#     #
#     # F.ProcessScans(50,70)
#
#         F.GetContactDelaysOnLine(np.array(xp[j]),np.array(yp[j]),6.3)
#
#         g = griddata((x,y), G, (xp,yp), method='nearest')
#
#         IL.append(abs(F.ApplyTFM(ind[j],OnLine=True))*g)
#
#     I.append(IL)
#
# IS = [[sum(III>=0.5*np.max(III))*dm for III in II] for II in I]
#
# pickle.dump({'IntensitySpread':IS, 'x':xx, 'y':yy}, open(pth+'CalBlockCoordinatesandIntesitySpreads.p','wb'))
#
#
# X = [xx[0] + xx[1], xx[2] + xx[3], xx[4] + xx[5]]
# Y = [yy[0] + yy[1], yy[2] + yy[3], yy[4] + yy[5]]]
# S = [IS[0] + IS[1], IS[2] + IS[3], IS[4] + IS[5]]
#
# xs = xx[6] + xx[7]
# ys = yy[6] + yy[7]
# ss = IS[6] + IS[7]
# size = []
#
# for i in range(len(xs)):
#
#     s = np.zeros(3)
#
#     for j in range(len(X)):
#
#         s[j] = griddata((X[j],Y[j]),S[j],(xs(i),ys(i)), method='linear')
#
#     size.append(griddata(np.array([1,2,3]),s,ss(i), method = 'linear'))
