import FMC as f
import numpy as np
import _pickle as pickle
from scipy.signal import hilbert

def MeasureThickness(A,fs,N,p,h,angle,cw,cs,Th):

    h = h + 0.5*p*(N-1)*np.sin(angle*np.pi/180)

    T = int(np.round(fs*2*(h/cw + Th/cs)))

    W = int(np.round(fs*0.25*Th/cs))

    b = np.argmax(np.abs(hilbert(A[T-W:T+W]))) + T - W

    return 0.5*cs*(fs*b - 2*(h/cw))

def MeasureProbeOffset(A,fs,N,p,h,angle,cw,cs,Th):

    phi = angle

    phir = np.arcsin((cs/cw)*np.sin(phi))

    T = int(np.round(2*fs*((Th/cs*np.cos(phir)) + (h + 0.5*p*(N-1)*np.sin(phi))/(cw*np.cos(phi)))))

    W = int(np.round(0.15*T))

    b = np.argmax(np.abs(hilbert(A[T-W:T+W]))) + T - W

    L = cs*np.sin(phir)*(0.5*b*fs - (h + 0.5*p*(N-1)*np.sin(phi))/(cw*np.cos(phi)))

    return L + np.tan(phi)*(h + 0.5*p*(N-1)*np.sin(phi)) + np.cos(phi)*(0.5*p*(N-1))

## Wedge Parameters needs to be estimated before running the script
# WP = f.EstimateWedgeParameters(F.AScans[0],fs,h,cw,angle,p))
#
# h1 = WP['Height']
# angle1 = WP['Angle']

WedgeParameter = {'Height':20.,'Velocity': 1.48, 'Angle':21.}
ProbeParameter = {'NumberofElements': 32, 'Pitch': 0.6}
PieceInfo = {'Velocity':5.9, 'Thickness': 9.5}

N = ProbeParameter['NumberOfElements']
p = ProbeParameter['Pitch']

h = WedgeParameter['Height']
angle = WedgeParametr['Angle'] *np.pi/180
cw = WedgeParameter['Velocity']
fs = 25.

Th = 9.5
cs = 3.24
cl = 5.9

WeldCap = 18
Haz = 13

xres = 0.1
yres = 0.1

xstart = np.round(-0.5*(N-1)*p)
xend = -xstart
ystart = 0
yend = -Th

Scans = pickle.load(open('mnt/c/Users/mmarvasti/Desktop','rb'))

F = f.LinearCapture(fs,p,N,WedgeParameters=WedgeParameter)

Linear90Skew = [F.PlaneWaveSweep(i,-angle1,cw,range(N)) for i in range(len(Scans))]

Th90Skew = np.array([MeasureThickness(Linear90Skew[i],fs,N,p,h1,angle1,cw,cw,Th) for i in range(len(Scans))])

Linear270Skew = [F.PlaneWaveSweep(i,-angle,cw,np.flip(range(N:2*N)) for i in range(len(Scans))]

Th270Skew = np.array([MeasureThickness(Linear270Skew[i],fs,N,p,h,angle,cw,cw,Th) for i in range(len(Scans))])

PitchCatchSweep = [F.PlaneWaveSweep(i,-60,cw,np.flip(range(N:2*N)) for i in range(len(Scans))]

SkewOffsets = np.array([MeasureProbeOffset(PitchCatchSweep[i],fs,N,p,h,angle,cw,cs,Th) for in range(len(Scans))])

# DirectDirect shear
DirectDirectShear = []

for i in range(len(SkewOffsets)):

    F.SetRectangularGrid(xstart,xend,ystart,end,xres,yres,Offset=SkewOffsets[i])

    F.GetWedgeDelays(cs)

    DirectDirectShear.append(self.Delays)

DirectDirectCompression = []

for i in range(len(SkewOffsets)):

    F.SetRectangularGrid(xstart,xend,ystart,end,xres,yres,Offset=SkewOffsets[i])

    F.GetWedgeDelays(cl)

    DirectDirectCompression.append(self.Delays)

BackWallDirect = []

for i in range(len(SkewOffsets)):

    F.SetRectangularGrid(xstart,xend,ystart,end,xres,yres,Offset=SkewOffsets[i])

    F.GetWedgeDelays(cl)

    DirectDirectCompression.append(self.Delays)
