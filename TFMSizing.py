import numpy as np
import _pickle as pickle
import FMC
import os
import itertools as it
import matplotlib.pylab as plt
from scipy.interpolate import griddata


b = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoFiles/MoScans/0DegreeContactCapture.p','rb'))
d = FMC.EstimateProbeDelays(b['AScans'][0], 25., 0.6, 25.4, hfraction=0.25,c=6.4)[0]
d = 0.5*(np.amax(d) + np.amin(d))
d = d*np.ones((64,64))

yholes = np.array([6.35, 12.7, 19.05, 25.4, 31.75, 38.1, 68.65, 62.3, 55.95, 49.6, 43.25, 36.9])
Yholes = np. array([9.525, 22.225, 34.925, 65.475, 52.775, 40.075])
holesize = np.array([1, 1, 2, 2, 2, 2, 2, 2])

# Getting First Coordinate Candidates
# X = np.arange(-21,21,1)
# Y = np.arange(1,73,1)
#
# # pth = '/mnt/c/Users/mmarvasti/Desktop/MoFiles/MoScans/ContactCalibrationScans/'
# pth = '/mnt/c/Users/mmarvasti/Desktop/MoFiles/MoScans/ContactCalibrationScans/New/'
# Dir = os.listdir(pth)
#
# Amp = []
# XX = []
# YY = []
#
# for j in range(len(Dir)):
#
#     scan = pickle.load(open(pth + Dir[j],'rb'))
#
#     # if (j<6):
#     #
#     #     F = FMC.LinearCapture(25,scan['AScans'][30:-30],0.6,64,d)
#     #
#     # else:
#     #
#     #     F = FMC.LinearCapture(25,scan['AScans'],0.6,64,d)
#
#     F = FMC.LinearCapture(25,scan['AScans'],0.6,64,d)
#
#     del(scan)
#
#     F.ProcessScans(50,70)
#
#     print('Processing Scan ' + str(j) + ' is Done')
#
#     F.GetContactDelays(X,Y,6.4)
#
#     I = [F.ApplyTFM(i) for i in range(len(F.AScans))]
#
#     print(str(j)+'Done')
#
#     x = np.zeros(len(I))
#     y = np.zeros(len(I))
#     amp = np.zeros(len(I))
#
#     for i in range(len(I)):
#
#         ind = np.unravel_index(np.argmax(np.abs(I[i])),I[i].shape)
#
#         F.GetContactDelays(np.arange(X[ind[1]]-5.,X[ind[1]]+5.,0.1),np.arange(Y[ind[0]]-5.,Y[ind[0]]+5.,0.1),6.4)
#
#         II = F.ApplyTFM(i)
#
#         iind = np.unravel_index(np.argmax(np.abs(II)), II.shape)
#
#         x[i] = F.xRange[iind[1]]
#         y[i] = F.yRange[iind[0]]
#         amp[i] = np.abs(II[iind[0],iind[1]])
#
#         print(str(i)+' ' + str(j))
#
#     Amp.append(amp)
#     XX.append(x)
#     YY.append(y)
#
#     del(F)
#
# pickle.dump({'Amplitude':Amp, 'x':XX, 'y':YY}, open(pth+'CalBlockCoordinatesandAmplitudesNew.p','wb'))

# Filtering Coordinates
# pth = '/mnt/c/Users/mmarvasti/Desktop/MoFiles/MoScans/ContactCalibrationScans/'
# data = pickle.load(open(pth + 'RawCoordinatsAmplitudes.p' ,'rb'))
#
# XX = data['x']
# YY = data['y']
# Amp = data['Amplitudes']
#
#
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
#             yi.append(YY[i][j])
#             amp.append(Amp[i][j])
#             indi.append(j)
#
#     xx.append(xi)
#     yy.append(yi)
#     amps.append(amp)
#     ind.append(indi)
#
# # pickle.dump({'Amplitude':amps, 'x':xx, 'y':yy}, open(pth+'CalBlockCoordinatesandAmplitudesFiltered.p','wb'))

# Getting Intensity Spreads on Coordinates
# x = xx[0] + xx[1]
# y = yy[0] + yy[1]
# Amps = amps[0] + amps[1]
# G = list(80/array(Amps))
#
# Correction = []
# CorrectedAmplitudes = []
#
# for i in range(len(xx)):
#
#     g = griddata((np.array(x),np.array(y)),np.array(G),(np.array(xx[i]),np.array(yy[i])),method='nearest')
#     Correction.append(g)
#     CorrectedAmplitudes.append(array(amps[i])*g)
#
# # pickle.dump({'Amplitude':amps, 'x':xx, 'y':yy, 'Correction': Correction, 'CorrectedAplitudes': CorrectedAmplitudes}, open(pth+'CalibrationData.p','wb'))
#
# pth = '/mnt/c/Users/mmarvasti/Desktop/MoFiles/MoScans/ContactCalibrationScans/New2/'
# Dir = os.listdir(pth)
#
# dm = 0.1
# m = np.arange(-4,4,dm)
#
# I = []
#
# for i in range(len(Dir)):
#
#     IL = []
#     scan = pickle.load(open(pth + Dir[i],'rb'))
#
#     # xp = []
#     # yp = []
#
#     for j in range(len(xx[i])):
#
#         phi = np.arctan(xx[i][j]/yy[i][j])
#         xpi = xx[i][j] + m*np.cos(phi)
#         ypi = yy[i][j] - m*np.sin(phi)
#
#         xp = []
#         yp = []
#
#         for k in range(len(xpi)):
#             if ((xpi[k]>-19.2)&(xpi[k]<19.2)&(ypi[k]>0)&(ypi[k]<75)):
#                 xp.append(xpi[k])
#                 yp.append(ypi[k])
#
#         F = FMC.LinearCapture(25,[scan['AScans'][ind[i][j]]],0.6,64,d)
#
#         F.ProcessScans(50,70)
#
#         F.GetContactDelaysOnLine(np.array(xp),np.array(yp),6.4)
#
#         g = griddata((np.array(x),np.array(y)),np.array(G),(np.array(xp),np.array(yp)),method='nearest')
#
#         # IL.append(abs(F.ApplyTFM(ind[i][j],OnLine=True))*g)
#         IL.append(abs(F.ApplyTFM(0,OnLine=True))*g)
#
#         del(F)
#
#         print(str(i) + ' ' + str(j) + ' Done')
#
#     del(scan)
#
#     I.append(IL)
#
# IS = [[sum(III>=0.5*np.max(III))*dm for III in II] for II in I]
#
# pickle.dump({'IntensitySpread':IS, 'x':xx, 'y':yy}, open(pth+'CalBlockIntesitySpreads.p','wb'))

#For Final Size interpolation
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


caldata = pickle.load(open('/mnt/c/Users/mmarvasti/MoFiles/MoScans/ContactCalibrationScans/CalibrationData.p','rb'))
xc = caldata['x']
yc = caldata['y']
g = caldata['Correction']

Xc = np.array(xc[0] + xc[1])
Yc = np.array(yc[0] + yc[1])
G = np.array(list(g[0])+list(g[1]))

dm = 0.1
m = np.arange(-4,4,dm)

yy1 = 9.525
yy2 = 22.225
yy3 = 34.925

X = np.arange(-21,21,0.5)
Y1 = np.arange(yy1-4,yy1+4,0.5)
Y2 = np.arange(yy2-4,yy2+4,0.5)
Y3 = np.arange(yy3-4,yy3+4,0.5)

pth = '/mnt/c/Users/mmarvasti/Desktop/MoFiles/MoScans/ContactCalibrationScans/ValidationScans/'

scan = pickle.load(open(pth + '4mmCalScan0Degree.p','rb'))

F = FMC.LinearCapture(25,[scan['AScans'][0]],0.6,64)

F.GetContactDelays(X,Y1,6.4)
delay1 = F.Delays

F.GetContactDelays(X,Y2,6.4)
delay2 = F.Delays

F.GetContactDelays(X,Y3,6.4)
delay3 = F.Delays

del(F)

N = len(scan['AScans'])
x1 = []
y1 = []
Is1 = []

x2 = []
y2 = []
Is2 = []

x3 = []
y3 = []
Is3 = []

for i in range(len(scan['AScans'])):

    F = FMC.LinearCapture(25,[scan['AScans'][i]],0.6,6.4,d)

    F.ProcessScans(50,70)

    F.Delays = delay1

    I = F.ApplyTFM(0)

    ind = np.unravel_index(np.argmax(np.abs(I),I.shape)

    F.GetContactDelays(np.arange(X[ind[1]]-4.,X[ind[1]]+4.,0.1),np.arange(Y1[ind[0]]-4.,Y1[ind[0]]+4.,0.1),6.4)

    I = F.ApplyTFM(0)

    ind = np.unravel_index(np.argmax(np.abs(I)), I.shape)

    if (abs(F.yRange[ind[0]] - yy1) < 2):

        x1.append(F.xRange[ind[1]])
        y1.append(F.yRange[ind[0]])

        phi = np.arctan(F.xRange[ind[1]]/F.yRange[ind[0]])
        xpi = F.xRange[ind[1]] + m*np.cos(phi)
        ypi = F.yRange[ind[0]] - m*np.sin(phi)

        xp = []
        yp = []

        for k in range(len(xpi)):
            if ((xpi[k]>-19.2)&(xpi[k]<19.2)&(ypi[k]>0)&(ypi[k]<75)):
                xp.append(xpi[k])
                yp.append(ypi[k])

        F.GetContactDelaysOnLine(np.array(xp),np.array(yp),6.4)

        g = griddata((Xc,Yc),G,(np.array(xp),np.array(yp)),method='nearest')

        I = abs(F.ApplyTFM(0,OnLine=True))*g

        Is1.append(np.sum(I>=0.5*np.max(I)))

    print( str(i) + ' 1 Done')


    F.Delays = delay2

    I = F.ApplyTFM(0)

    ind = np.unravel_index(np.argmax(np.abs(I),I.shape)

    F.GetContactDelays(np.arange(X[ind[1]]-4,X[ind[1]]+4,0.1),np.arange(Y2[ind[0]]-4,Y2[ind[0]]+4,0.1),6.4)

    I = F.ApplyTFM(0)

    ind = np.unravel_index(np.argmax(np.abs(I)), I.shape)

    if (abs(F.yRange[ind[0]] - yy2) < 2):

        x2.append(F.xRange[ind[1]])
        y2.append(F.yRange[ind[0]])

        phi = np.arctan(F.xRange[ind[1]]/F.yRange[ind[0]])
        xpi = F.xRange[ind[1]] + m*np.cos(phi)
        ypi = F.yRange[ind[0]] - m*np.sin(phi)

        xp = []
        yp = []

        for k in range(len(xpi)):
            if ((xpi[k]>-19.2)&(xpi[k]<19.2)&(ypi[k]>0)&(ypi[k]<75)):
                xp.append(xpi[k])
                yp.append(ypi[k])

        F.GetContactDelaysOnLine(np.array(xp),np.array(yp),6.4)

        g = griddata((Xc,Yc),G,(np.array(xp),np.array(yp)),method='nearest')

        I = abs(F.ApplyTFM(0,OnLine=True))*g

        Is2.append(np.sum(I>=0.5*np.max(I)))

    print( str(i) + ' 2 Done')


    F.Delays = delay3

    I = F.ApplyTFM(0)

    ind = np.unravel_index(np.argmax(np.abs(I),I.shape)

    F.GetContactDelays(np.arange(X[ind[1]]-4,X[ind[1]]+4,0.1),np.arange(Y3[ind[0]]-4,Y3[ind[0]]+4,0.1),6.4)

    I = F.ApplyTFM(0)

    ind = np.unravel_index(np.argmax(np.abs(I)), I.shape)

    if (abs(F.yRange[ind[0]] - yy3) < 2):

        x3.append(F.xRange[ind[1]])
        y3.append(F.yRange[ind[0]])

        phi = np.arctan(F.xRange[ind[1]]/F.yRange[ind[0]])
        xpi = F.xRange[ind[1]] + m*np.cos(phi)
        ypi = F.yRange[ind[0]] - m*np.sin(phi)

        xp = []
        yp = []

        for k in range(len(xpi)):
            if ((xpi[k]>-19.2)&(xpi[k]<19.2)&(ypi[k]>0)&(ypi[k]<75)):
                xp.append(xpi[k])
                yp.append(ypi[k])

        F.GetContactDelaysOnLine(np.array(xp),np.array(yp),6.4)

        g = griddata((Xc,Yc),G,(np.array(xp),np.array(yp)),method='nearest')

        I = abs(F.ApplyTFM(0,OnLine=True))*g

        Is3.append(np.sum(I>=0.5*np.max(I)))

    print( str(i) + ' 3 Done')

    del(F)


pickle.dump({'x':[x1,x2,x3], 'y':[y1,y2,y3], 'IntensitySpread':[Is1,Is2,Is3]}, open(pth+'4mmCalScan0DegreeValidationData.p','wb'))

del(scan)


yy1 = 65.475
yy2 = 52.775
yy3 = 40.075

X = np.arange(-21,21,0.5)
Y1 = np.arange(yy1-4,yy1+4,0.5)
Y2 = np.arange(yy2-4,yy2+4,0.5)
Y3 = np.arange(yy3-4,yy3+4,0.5)

scan = pickle.load(open(pth + '4mmCalScan0DegreeOPP.p','rb'))

F = FMC.LinearCapture(25,[scan['AScans'][0]],0.6,64)

F.GetContactDelays(X,Y1,6.4)
delay1 = F.Delays

F.GetContactDelays(X,Y2,6.4)
delay2 = F.Delays

F.GetContactDelays(X,Y3,6.4)
delay3 = F.Delays

del(F)

N = len(scan['AScans'])
x1 = []
y1 = []
Is1 = []

x2 = []
y2 = []
Is2 = []

x3 = []
y3 = []
Is3 =[]

for i in range(len(scan['AScans'])):

    F = FMC.LinearCapture(25,[scan['AScans'][i]],0.6,6.4,d)

    F.ProcessScans(50,70)

    F.Delays = delay1

    I = F.ApplyTFM(0)

    ind = np.unravel_index(np.argmax(np.abs(I),I.shape)

    F.GetContactDelays(np.arange(X[ind[1]]-4.,X[ind[1]]+4.,0.1),np.arange(Y1[ind[0]]-4.,Y1[ind[0]]+4.,0.1),6.4)

    I = F.ApplyTFM(0)

    ind = np.unravel_index(np.argmax(np.abs(I)), I.shape)

    if (abs(F.yRange[ind[0]] - yy1) < 2):

        x1.append(F.xRange[ind[1]])
        y1.append(F.yRange[ind[0]])

        phi = np.arctan(F.xRange[ind[1]]/F.yRange[ind[0]])
        xpi = F.xRange[ind[1]] + m*np.cos(phi)
        ypi = F.yRange[ind[0]] - m*np.sin(phi)

        xp = []
        yp = []

        for k in range(len(xpi)):
            if ((xpi[k]>-19.2)&(xpi[k]<19.2)&(ypi[k]>0)&(ypi[k]<75)):
                xp.append(xpi[k])
                yp.append(ypi[k])

        F.GetContactDelaysOnLine(np.array(xp),np.array(yp),6.4)

        g = griddata((Xc,Yc),G,(np.array(xp),np.array(yp)),method='nearest')

        I = abs(F.ApplyTFM(0,OnLine=True))*g

        Is1.append(np.sum(I>=0.5*np.max(I)))

    print( str(i) + ' 1 Done')


    F.Delays = delay2

    I = F.ApplyTFM(0)

    ind = np.unravel_index(np.argmax(np.abs(I),I.shape)

    F.GetContactDelays(np.arange(X[ind[1]]-4,X[ind[1]]+4,0.1),np.arange(Y2[ind[0]]-4,Y2[ind[0]]+4,0.1),6.4)

    I = F.ApplyTFM(0)

    ind = np.unravel_index(np.argmax(np.abs(I)), I.shape)

    if (abs(F.yRange[ind[0]] - yy2) < 2):

        x2.append(F.xRange[ind[1]])
        y2.append(F.yRange[ind[0]])

        phi = np.arctan(F.xRange[ind[1]]/F.yRange[ind[0]])
        xpi = F.xRange[ind[1]] + m*np.cos(phi)
        ypi = F.yRange[ind[0]] - m*np.sin(phi)

        xp = []
        yp = []

        for k in range(len(xpi)):
            if ((xpi[k]>-19.2)&(xpi[k]<19.2)&(ypi[k]>0)&(ypi[k]<75)):
                xp.append(xpi[k])
                yp.append(ypi[k])

        F.GetContactDelaysOnLine(np.array(xp),np.array(yp),6.4)

        g = griddata((Xc,Yc),G,(np.array(xp),np.array(yp)),method='nearest')

        I = abs(F.ApplyTFM(0,OnLine=True))*g

        Is2.append(np.sum(I>=0.5*np.max(I)))

    print( str(i) + ' 2 Done')


    F.Delays = delay3

    I = F.ApplyTFM(0)

    ind = np.unravel_index(np.argmax(np.abs(I),I.shape)

    F.GetContactDelays(np.arange(X[ind[1]]-4,X[ind[1]]+4,0.1),np.arange(Y3[ind[0]]-4,Y3[ind[0]]+4,0.1),6.4)

    I = F.ApplyTFM(0)

    ind = np.unravel_index(np.argmax(np.abs(I)), I.shape)

    if (abs(F.yRange[ind[0]] - yy3) < 2):

        x3.append(F.xRange[ind[1]])
        y3.append(F.yRange[ind[0]])

        phi = np.arctan(F.xRange[ind[1]]/F.yRange[ind[0]])
        xpi = F.xRange[ind[1]] + m*np.cos(phi)
        ypi = F.yRange[ind[0]] - m*np.sin(phi)

        xp = []
        yp = []

        for k in range(len(xpi)):
            if ((xpi[k]>-19.2)&(xpi[k]<19.2)&(ypi[k]>0)&(ypi[k]<75)):
                xp.append(xpi[k])
                yp.append(ypi[k])

        F.GetContactDelaysOnLine(np.array(xp),np.array(yp),6.4)

        g = griddata((Xc,Yc),G,(np.array(xp),np.array(yp)),method='nearest')

        I = abs(F.ApplyTFM(0,OnLine=True))*g

        Is3.append(np.sum(I>=0.5*np.max(I)))

    print( str(i) + ' 3 Done')

    del(F)

pickle.dump({'x':[x1,x2,x3], 'y':[y1,y2,y3], 'IntensitySpread':[Is1,Is2,Is3]}, open(pth+'4mmCalScan0DegreeOPPValidationData.p','wb'))

del(scan)
