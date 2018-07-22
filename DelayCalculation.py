import FMC as f
import numpy as np
import _pickle as pickle
from scipy.signal import hilbert
import itertools
import matplotlib.pylab as plt
from imp import reload

# For SegmentWedge:

reload(f)

fs, cw, Th, cs, cl, p, h, angle, N = 25., 2.33, 10, 3.24, 5.9, 0.6, 5.79, 31, 32

xres, yres = 0.1, 0.1

xstart, xend, ystart, yend, Offset = 0, 25, 0, 15, 0

WedgeParameter = {'Height':h,'Velocity': cw, 'Angle':angle}

Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/CheckBackWallDelays/BackwallTest.p','rb'))

F = f.LinearCapture(fs,[Scans['AScans'][0]],p,32,WedgeParameters=WedgeParameter)

# plt.plot(Scans['AScans'][0][0,0,:])

F.ProcessScans(100,100)

# plt.plot(F.AScans[0][0,0,:])
#
# plt.plot(F.AScans[0][0,31,:])
#
# plt.plot(F.AScans[0][5,25,:])
# plt.show()

F.SetRectangularGrid(xstart,xend,ystart,yend,xres,yres)

F.GetWedgeDelays(cs,Offset)

II = F.ApplyTFM(0)

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
# xstart, xend, ystart, yend, Offset = 10, 50, 5, Th-yres, 0
#
# WedgeParameter = {'Height':h,'Velocity': cw, 'Angle':angle}
#
# Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/Sample11.p','rb'))
#
# F = f.LinearCapture(fs,[Scans['AScans'][0]],p,64,WedgeParameters=WedgeParameter)
#
# F.ProcessScans(100,100)
#
# F.KeepElements(range(N))
#
# F.SetRectangularGrid(xstart,xend,ystart,yend,xres,yres)
#
# F.GetWedgeDelays(cs,Offset)
#
# F.NumberOfElements = 32
#
# # Direct = F.Delays
#
# # F.GetWedgeBackwallDelays((cs,cs),Th,Offset)
# #
# # BackWall = F.Delays
# #
# # F.Delays = (Direct[0],Backwall[0])
#
# I = F.ApplyTFM(0)
