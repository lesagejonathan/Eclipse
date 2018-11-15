import numpy as np
import os
import FMC
import _pickle as pickle

pth = '/mnt/c/Users/user/Documents/FeederFMC/MJData/'

fls = os.listdir(pth)

fls = [fl for fl in fls if fl.endswith('.p')]

for fl in fls:

    a = pickle.load(open(pth+fl,'rb'))

    F = FMC.LinearCapture(25.,a['AScans'],0.5,32)

    F.ProcessScans(50,80)

    A = [F.PlaneWaveSweep(i,)
