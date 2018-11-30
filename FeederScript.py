import os
import FMC
import numpy as np
import matplotlib.pylab as plt

pth = '/mnt/c/Users/mmarvasti/Desktop/MCR'

files = os.listdir(pth)

cL = 6.041
cs = 3.279

Elements1 = list(range(16))
Elements2 = list(range(16,32))

for i in range(len(files)):

    scan = pickle.load(open(pth + '/' + files[i],'rb'))

    F = FMC.LinearCapture(25, scan['AScans'], 0.5, 32, WedgeParameters = {'Velocity':2.33, 'Height':2.9, 'Angle':39})

    F.ProcessScans(40,40)

    WA90 = []
    H90 = []
    Th90 = []
    WA270 = []
    H270 = []
    Th270 = []
    Offset = []

    for j in range(len(F.AScans)):

        WA90.append(F.EstimateSweepBasedWedgeAngle(Elements1, j))
        H90.append(F.EstimateWedgeHeight(Elements1, j))
        Th90.append(F.EstimateThickness(cL, Elements1, WA90[j], j))

        WA270.append(F.EstimateSweepBasedWedgeAngle(Elements2,j))
        H270.append(F.EstimateWedgeHeight(Elements2,j))
        Th270.append(F.EstimateThickness(cL, Elements2, WA270[j], j))

        Offset.append(F.EstimateSweepBasedProbesOffset(0.5*(WA90[j]+WA270[j]), 0.5*(H90[j]+H270[j]), cs, 0.5*(Th90[j]+Th270[j]), 6, j))


    dict = {'WA90': np.array(WA90), 'H90': np.array(H90), 'Th90': np.array(Th90), 'WA270': np.array(WA270), 'H270': np.array(H270), 'Th270': np.array(Th270), 'Offset':Offset}
    pickle.dump(dict,open(pth + '/' + files[i][0:len(files[i])-2] + '_Info.p','wb'))


    fig = plt.figure()
    plt.plot(array(WA90))
    plt.plot(array(WA270))
    plt.savefig(pth + '/' + files[i][0:len(files[i])-2] + '_WedgeAngles.png')

    del(fig)

    fig = plt.figure()
    plt.plot(array(H90))
    plt.plot(array(H270))
    plt.savefig(pth + '/' + files[i][0:len(files[i])-2] + '_WedgeHeights.png')

    del(fig)

    fig = plt.figure()
    plt.plot(array(Th90))
    plt.plot(array(Th270))
    plt.savefig(pth + '/' + files[i][0:len(files[i])-2] + '_Thicknesses.png')

    del(fig)

    fig = plt.figure()
    plt.plot(array(Offset))
    plt.savefig(pth + '/' + files[i][0:len(files[i])-2] + '_ProbesOffset.png')

    del(fig)

del(plt)
#     plt.plot(WA90)
#     plt.plot(WA270)
#     savefig(pth + '/' + files[i][0:len(files[i])-2] + '_WedgeAngles.png')
#
#     plt.plot(WA90)
#     plt.plot(WA270)
#     savefig(pth + '/' + files[i][0:len(files[i])-2] + '_WedgeAngles.png')


    # dict = {}
    # dict['WedgeAngle90'] = []
    # dict['WedgeAngle270'] = []
    # dict['WedgeHeight90'] = []
    # dict['WedgeHeight270'] = []
