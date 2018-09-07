import FMC as f
import numpy as np
import _pickle as pickle
from scipy.signal import hilbert
import itertools
import matplotlib.pylab as plt
from imp import reload
import copy

# For SegmentWedge:

# reload(f)
#
# fs, cw, Th, cs, cl, p, h, angle, N = 25., 2.33, 10, 3.24, 5.9, 0.6, 5.79, 31, 32
#
# xres, yres = 0.1, 0.1
#
# xstart, xend, ystart, yend, Offset = 0, 25, 5, 12, 0
#
# WedgeParameter = {'Height':h,'Velocity': cw, 'Angle':angle}
#
# Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/CheckBackWallDelays/BackwallTest.p','rb'))
#
# F = f.LinearCapture(fs,[Scans['AScans'][0]],p,32,WedgeParameters=WedgeParameter)
#
# F.ProcessScans(100,100)
#
# F.SetRectangularGrid(xstart,xend,ystart,yend,xres,yres)
#
# F.GetWedgeDelays(cs,Offset)
#
# II = F.ApplyTFM(0)
#
# Direct = F.Delays
#
# F.GetWedgeBackwallDelays((cs,cs),Th,Offset)
#
# BackWall = F.Delays
#
# F.Delays = (BackWall[0],Direct[0])
#
# I = F.ApplyTFM(0)





# For Husky:

# reload(f)
#
# fs, cw, Th, cs, cl, p, h, angle, N = 25., 1.48, 10, 3.24, 5.9, 0.6, 32.8, 13, 32
#
# xres, yres = 0.1, 0.1
#
# # xstart, xend, ystart, yend, Offset = 0, 80, 0, 25, 0
#
# xstart, xend, ystart, yend, Offset = -26, 26, 0, 25, 36.6
#
# WedgeParameter = {'Height':h,'Velocity': cw, 'Angle':angle}
#
# Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/Sample11.p','rb'))
#
# F = f.LinearCapture(fs,[Scans['AScans'][0]],p,64,WedgeParameters=WedgeParameter)
#
# F.ProcessScans(100,100)
#
# # F.KeepElements(range(N))
#
# F.SetRectangularGrid(xstart,xend,ystart,yend,xres,yres)
#
# # delays = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/HuskyDelaysNoOffset.p','rb'))
# delays = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/HuskyImages/HuskyDelaysOffset36p6.p','rb'))
#
# Elements = (range(32), range(32,32+16))
#
# TFMImages = {}
#
# d = delays['SDirect'][0]
# dd=[]
#
# for n in range(len(d)):
#
#     dd.append(np.flip(d[n],axis=1))
#
# F.Delays = (d,dd)
# # F.Delays = (delays['SDirect'][0],list(np.flip(delays['SDirect'][0],axis=0)))
#
# TFMImages['SPitchCatch'] = F.ApplyTFM(0,Elements)
#
# print('First')
#
# d = delays['LDirect'][0]
# dd=[]
#
# for n in range(len(d)):
#
#     dd.append(np.flip(d[n],axis=1))
#
# F.Delays = (d,dd)
#
# # F.Delays = (delays['LDirect'][0],list(np.flip(delays['LDirect'][0],axis=0)))
#
# TFMImages['LPitchCatch'] = F.ApplyTFM(0,Elements)
#
# print('Second')


# For Husky:

reload(f)

fs, cw, Th, cs, cl, p, h, angle, N = 25., 1.48, 10, 3.24, 5.9, 0.6, 32.8, 13, 32

xres, yres = 0.1, 0.1

# xstart, xend, ystart, yend, Offset = 0, 80, 0, 25, 0

xstart, xend, ystart, yend, Offset = -26, 26, 0, 25, 36.6

WedgeParameter = {'Height':h,'Velocity': cw, 'Angle':angle}

Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/Sample11.p','rb'))

F = f.LinearCapture(fs,[Scans['AScans'][0]],p,64,WedgeParameters=WedgeParameter)

F.ProcessScans(100,100)

# F.KeepElements(range(N))

F.SetRectangularGrid(xstart,xend,ystart,yend,xres,yres)

# delays = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/HuskyDelaysNoOffset.p','rb'))
delays = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/HuskyImages/HuskyDelaysOffset36p6.p','rb'))

Elements = (range(32), range(32,32+16))

TFMImages = {}

d = delays['SDirect'][0]
dd=[]

for n in range(len(d)):

    dd.append(np.flip(d[n],axis=1))

F.Delays = (d,dd)
# F.Delays = (delays['SDirect'][0],list(np.flip(delays['SDirect'][0],axis=0)))

TFMImages['SPitchCatch'] = F.ApplyTFM(0,Elements)

print('First')

d = delays['LDirect'][0]
dd=[]

for n in range(len(d)):

    dd.append(np.flip(d[n],axis=1))

F.Delays = (d,dd)

# F.Delays = (delays['LDirect'][0],list(np.flip(delays['LDirect'][0],axis=0)))

TFMImages['LPitchCatch'] = F.ApplyTFM(0,Elements)

print('Second')
