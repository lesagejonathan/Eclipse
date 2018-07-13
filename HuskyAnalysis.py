import FMC as f
import numpy as np
import _pickle as pickle
from scipy.signal import hilbert
import itertools
import matplotlib.pylab as plt

from imp import reload

reload(f)

def MeasureThickness(AScans,fs,N,p,h,angle,cw,cl,Th):

    # h = h + 0.5*p*(N-1)*np.sin(angle*np.pi/180)

    h = h + 0.5*p*(N-1)*np.sin(angle)

    print(h)

    T = int(np.round(fs*2*(h/cw + Th/cl)))

    print(T)

    W = int(np.round(fs*0.25*Th/cl))

    print(W)

    b = np.argmax(np.abs(hilbert(np.real(AScans[0,0,T-W:T+W])))) + T - W

    print(b)

    plt.plot(AScans[0,0,:])
    plt.show()

    return 0.5*cl*(b/fs - 2*(h/cw))

def MeasureProbeOffset(AScans,SweepAngle,fs,N,p,h,angle,cw,cs,Th):

    phiw = angle
    a = SweepAngle

    phir = np.arcsin((cs/cw)*np.sin(phiw+a))

    T = int(np.round(2*fs*(Th/cs*np.cos(phir) + (h + 0.5*p*(N-1)*np.sin(phiw))/(cw*np.cos(phiw+a)))))

    W = int(np.round(0.15*T))

    s = np.abs(hilbert(np.real(AScans[T-W:T+W])))

    b = np.argmax(s) + T - W

    L = cs*np.sin(phir)*(0.5*b/fs - (h + 0.5*p*(N-1)*np.sin(phiw))/(cw*np.cos(phiw+a)))

    offset = L + np.tan(phiw+a)*(h + 0.5*p*(N-1)*np.sin(phiw)) + np.cos(phiw)*(0.5*p*(N-1))

    return (s,offset)

def KeepDiagonalElements(AScans):

    return np.array([AScans[n,n,:] for n in range(AScans.shape[0])])

def EstimateWedgeParameters(AScans,fs,h,cw,angle,p):

    from scipy.signal import hilbert

    A = KeepDiagonalElements(AScans)

    print(str(A.shape[0]))

    B = np.zeros(A.shape[0]-3)

    for i in range(A.shape[0]-3):

        eh = h + i*p*np.sin(angle*np.pi/180)

        t = int(np.round(fs*2*eh/cw))

        w = int(np.round(fs*0.25*eh/cw))

        b = np.argmax(np.abs(hilbert(np.real(A[i,t-w:t+w])))) + t - w

        B[i] = (0.5*cw*b)/fs

    Co = np.polyfit(np.array(range(A.shape[0]-3)),B,1)

    # plt.plot(B)
    # plt.show()

    return {'Height':Co[1], 'Angle':np.arcsin(Co[0]/p)*180/np.pi}

# fs, cw, T, cs, cl, p, h, angle, N = 25., 1.48, 9.5, 3.24, 5.9, 0.6, 20., 21., 32

# Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/Husky/HuskyScan2.p','rb'))

# fs, cw, T, cs, cl, p, h, angle, N = 25., 1.48, 25.4, 3.24, 5.9, 0.6, 32, 13, 32

Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/LCalibration.p','rb'))

fs, cw, T, cs, cl, p, h, angle, N = 25., 1.48, 10, 3.24, 5.9, 0.6, 32, 13, 32

WedgeParameter = {'Height':h ,'Velocity': cw, 'Angle':angle}

F = f.LinearCapture(fs,[Scans['AScans'][0]],p,64,WedgeParameters=WedgeParameter)

F.ProcessScans(100,100)





F.KeepElements(range(N))

# # A = KeepDiagonalElements(F.AScans[0])
#
# WP = EstimateWedgeParameters(F.AScans[0],fs,h,cw,angle,p)
#
# WedgeParameter = {'Height':WP['Height'],'Velocity': cw, 'Angle':WP['Angle']}
#
# h = WedgeParameter['Height']
# angle = WedgeParameter['Angle'] *np.pi/180
# cw = WedgeParameter['Velocity']
# plt.plot(F.AScans[0][15,15,:])
# plt.show()
# B = F.PlaneWaveSweep(0,(np.array([-angle]),np.array([-angle])),(range(N),range(N)),cw)
# plt.plot(B[0,0,:])
# plt.show()

# Th = MeasureThickness(B,fs,N,p,h,angle,cw,cl,T)

# del (F)
#
# F = f.LinearCapture(fs,[Scans['AScans'][0]],p,64,WedgeParameters=WedgeParameter)
#
# F.ProcessScans(100,100)
#
# sweepangles = np.arange(-15,15,0.5) * np.pi/180
#
# PC = F.PlaneWaveSweep(0,(sweepangles,sweepangles),(np.array(range(32)),np.array(range(32,64))), cw)
#
# S = np.array([MeasureProbeOffset(PC[i,i,:],sweepangles[i],fs,N,p,h,angle,cw,cs,Th)[0] for i in range(len(PC))])
#
# m = np.argmax(S)
#
# Offset = MeasureProbeOffset(PC[m,m,:],sweepangles[m],fs,N,p,h,angle,cw,cs,Th)[1]
#
# Offset = (37.5,37.5)

WeldCap, Haz = 16, 2

xres = 0.25
yres = 0.25

xstart = -0.5 * (WeldCap + Haz)
xend = -xstart
ystart = 0
yend = Th

F.SetRectangularGrid(xstart,xend,ystart,yend,xres,yres)

HuskyDelays = {}

c= (cl,cl)

Offset = (35,35)

HuskyDelays['DirectDirect'] = F.GetDelays['DirectDirect'](c,Offset)



# c = list(itertools.product([cs,cl],repeat=2))
 HuskyDelays['DirectDirect'] = [(c[i], F.GetDelays['DirectDirect'](c[i],Offset)) for i in range(len(c))]
#
# c = list(itertools.product([cs,cl],repeat=3))
# HuskyDelays['BackWallDirect'] = [(c[i], F.GetDelays['BackwallDirect'](c[i],Th,Offset)) for i in range(len(c))]
#
# c = list(itertools.product([cs,cl],repeat=4))
# HuskyDelays['BackwallBackwall'] = [(c[i], F.GetDelays['BackwallBackwall'](c[i],Th,Offset)) for i in range(len(c))]
