import FMC as f
import numpy as np
import _pickle as pickle
from scipy.signal import hilbert
import itertools

def MeasureThickness(AScans,fs,N,p,h,angle,cw,cs,Th):

    h = h + 0.5*p*(N-1)*np.sin(angle*np.pi/180)

    T = int(np.round(fs*2*(h/cw + Th/cs)))

    W = int(np.round(fs*0.25*Th/cs))

    b = np.argmax(np.abs(hilbert(AScans[T-W:T+W]))) + T - W

    return 0.5*cs*(fs*b - 2*(h/cw))

def MeasureProbeOffset(AScans,fs,N,p,h,angle,cw,cs,Th):

    phi = angle

    phir = np.arcsin((cs/cw)*np.sin(phi))

    T = int(np.round(2*fs*((Th/cs*np.cos(phir)) + (h + 0.5*p*(N-1)*np.sin(phi))/(cw*np.cos(phi)))))

    W = int(np.round(0.15*T))

    b = np.argmax(np.abs(hilbert(AScans[T-W:T+W]))) + T - W

    L = cs*np.sin(phir)*(0.5*b*fs - (h + 0.5*p*(N-1)*np.sin(phi))/(cw*np.cos(phi)))

    return L + np.tan(phi)*(h + 0.5*p*(N-1)*np.sin(phi)) + np.cos(phi)*(0.5*p*(N-1))

def KeepDiagonalElements(AScans):

    N = len(AScans)

    A = np.zeros((len(AScans),len(AScans[0][0])))

    for i in range(len(AScans)):

        A[i] = AScan[i][i]

    return A

def EstimateWedgeParameters(AScans,fs,h,cw,angle,p):

    from scipy.signal import hilbert

    A = KeepDiagonalElements(AScans)

    B = np.zeros(len(A))

    for i in range(len(A)):

        h = h + i*p*np.sin(angle*np.pi/180)

        t = int(np.round(fs*2*h/cw))

        w = int(np.round(fs*0.25*h/cw))

        b = np.argmax(np.abs(hilbert(A[i][t-w:t+w]))) + t - w

        B[i] = 0.5*cw*fs*b

    c = np.polyfit(np.array(range(len(A))),B,1)

    return {'Height':c[1], 'Angle':np.arcsin(c[0]/p)*180/mp.pi)

fs, cw, T, cs, cl, p, h, angle, N = 25., 1.48, 9.5, 3.24, 5.9, 0.6, 20., 21., 32.

Scans = pickle.load(open('mnt/c/Users/mmarvasti/Desktop','rb'))

WedgeParameter = {'Height':h ,'Velocity': cw, 'Angle':angle}

F = f.LinearCapture(fs,Scans[0],p,N,WedgeParameters=WedgeParameter)

F.KeepElements(range(N))

WP = EstimateWedgeParameters(F.AScans,fs,h,cw,angle,p)

WedgeParameter = {'Height':WP['Height'],'Velocity': cw, 'Angle':WP['Angle']}

h = WedgeParameter['Height']
angle = WedgeParametr['Angle'] *np.pi/180
cw = WedgeParameter['Velocity']

Th = MeasureThickness(Scans[0],fs,N,p,h,angle,cw,cs,T)

F = F.LinearCapture(fs,Scans[0],p,N,WedgeParameters=WedgeParameter)

del(F)

AScan = F.PlaneWaveSweep(0, 0, (np.array(range(32)),np.array(range(32,64))), cw)

Offset = MeasureProbeOffset(AScan,fs,N,p,h,angle,cw,cs,Th)

WeldCap, Haz = 16, 2

xres = 0.1
yres = 0.1

xstart = -0.5 * (WeldCap + Haz)
xend = -xstart
ystart = 0
yend = -Th

F.SetRectangularGrid(xstart,xend,ystart,yend,xres,yres)

HuskyDelays = {}

c = list(itertools.product([cs,cl],repeat=2))
HuskyDelays['DirectDirect'] = [(c[i], F.GetDelays['DirectDirect'](c[i],Offset)) for i in range(len(c))]

c = list(itertools.product([cs,cl],repeat=3))
HuskyDelays['BackWallDirect'] = [(c[i], F.GetDelays['BackwallDirect'](c[i],Th,Offset)) for i in range(len(c))]

c = list(itertools.product([cs,cl],repeat=4))
HuskyDelays['BackwallBackwall'] = [(c[i], F.GetDelays['BackwallBackwall'](c[i],Th,Offset)) for i in range(len(c))]
