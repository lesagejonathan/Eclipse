import FMC as f
import numpy as np
import _pickle as pickle
from scipy.signal import hilbert
import itertools
import matplotlib.pylab as plt
from imp import reload

reload(f)

fs, cw, cs, p, h, angle, N = 25., 2.978, 3.24, 0.6, 27, 0, 64

Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/LCalibration.p','rb'))

WedgeParameter = {'Height':h ,'Velocity': cw, 'Angle':angle*np.pi/180}

F = f.LinearCapture(fs,[Scans['AScans'][0]],p,64,WedgeParameters=WedgeParameter)

xres = 0.1
yres = 0.1

xstart = -5
xend = (N-1)*p + 5
ystart = 0
yend = 100
Ofset = 0

F.SetRectangularGrid(xstart,xend,ystart,yend,xres,yres)

HarfangBlockDelays = {}

HarfangBlockDelays['ZeroDegreeWedge'] = F.GetDelays('Direct',cs,Offset)

pickle.dump(HarfangBlockDelays,open('/mnt/c/Users/mmarvasti/Desktop/MoScans/HarfangBlock.p','wb'))
