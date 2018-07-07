import FMC as f
import numpy as np
import _pickle as pickle
from scipy.signal import hilbert

WedgeParameter = {'Height':20.,'Velocity': 1.48, 'Angle':21.}
ProbeParameter = {'NumberofElements': 32, 'Pitch': 0.6}
PieceInfo = {'Velocity':5.9, 'Thickness': 9.5}

N = ProbeParameter['NumberOfElements']
p = ProbeParameter['Pitch']

h = WedgeParameter['Height']
angle = WedgeParametr['Angle'] *np.pi/180
cw = WedgeParameter['Velocity']
fs = 25.

Scans = pickle.load(open('mnt/c/Users/mmarvasti/Desktop','rb'))

F = f.LinearCapture(fs,p,N,WedgeParameters=WedgeParameter)

F.KeepElements(range(N))

WP = f.EstimateWedgeParameters(F.AScans[0],fs,h,cw,angle,p))

h1 = WP['Height']
angle1 = WP['Angle']

A = F.PlaneWaveSweep(0,np.array([-angle]))
