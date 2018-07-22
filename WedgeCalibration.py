import FMC as f
import numpy as np
import _pickle as pickle
from scipy.signal import hilbert
import itertools
import matplotlib.pylab as plt
from imp import reload

reload(f)

def MeasureThickness(AScans,fs,N,p,h,angle,cw,cl,Th):

    h = h + 0.5*p*(N-1)*np.sin(angle)

    T = int(np.round(fs*2*(h/cw + Th/cl)))

    W = int(np.round(fs*0.25*Th/cl))

    b = np.argmax(np.abs(hilbert(np.real(AScans[0,0,T-W:T+W])))) + T - W

    t1 = b/fs

    T = int(np.round(fs*2*(h/cw + 2*Th/cl)))

    b = np.argmax(np.abs(hilbert(np.real(AScans[0,0,T-W:T+W])))) + T - W

    t2 = b/fs

    return 0.5*cl*(t2-t1)

    # return 0.5*cl*(b/fs - 2*(h/cw))

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

    return (np.max(s),offset)

def KeepDiagonalElements(AScans):

    return np.array([AScans[n,n,:] for n in range(AScans.shape[0])])

def EstimateWedgeParameters(AScans,fs,h,cw,angle,p):

    from scipy.signal import hilbert

    A = KeepDiagonalElements(AScans)

    B = np.zeros(A.shape[0])

    for i in range(A.shape[0]):

        eh = h + i*p*np.sin(angle*np.pi/180)

        t = int(np.round(fs*2*eh/cw))

        w = int(np.round(fs*0.25*eh/cw))

        b = np.argmax(np.abs(hilbert(np.real(A[i,t-w:t+w])))) + t - w

        B[i] = (0.5*cw*b)/fs

    plt.plot(B)
    plt.show()

    # NN=A.shape[0]-1

    Co = np.polyfit(np.array(range(A.shape[0])),B,1)

    return {'Height':Co[1], 'Angle':np.arcsin(Co[0]/p)*180/np.pi}

fs, cw, Th, cs, cl, p, h, angle, N = 25., 2.33, 10, 3.24, 5.9, 0.6, 6.7, 31.9, 32

WedgeParameter = {'Height':h,'Velocity': cw, 'Angle':angle}

Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/CheckBackWallDelays/BackwallTest.p','rb'))

F = f.LinearCapture(fs,[Scans['AScans'][1]],p,32,WedgeParameters=WedgeParameter)

F.ProcessScans(100,100)

# F.KeepElements(range(N))

WP = EstimateWedgeParameters(F.AScans[0],fs,h,cw,angle,p)

WedgeParameter = {'Height':WP['Height'],'Velocity': cw, 'Angle':WP['Angle']}

h = WedgeParameter['Height']
angle = WedgeParameter['Angle'] *np.pi/180
cw = WedgeParameter['Velocity']

B = F.PlaneWaveSweep(0,(np.array([-angle]),np.array([-angle])),(range(N),range(N)),cw)

Th = MeasureThickness(B,fs,N,p,h,angle,cw,cl,T)

print(h)
print(angle*180/np.pi)
print(Th)

del(F)

# F = f.LinearCapture(fs,[Scans['AScans'][0]],p,64,WedgeParameters=WedgeParameter)
#
# F.ProcessScans(100,100)
#
# sweepangles = np.arange(0,3,0.25) * np.pi/180
#
# PC = F.PlaneWaveSweep(0,sweepangles,(np.array(range(32)),np.array(range(32,64))), cw)
#
# S = np.array([MeasureProbeOffset(PC[i,:],sweepangles[i],fs,N,p,h,angle,cw,cs,Th)[0] for i in range(PC.shape[0])])
#
# m = np.argmax(S)
#
# Offset = MeasureProbeOffset(PC[m,:],sweepangles[m],fs,N,p,h,angle,cw,cs,Th)[1]
#
# print('Offset is:')
# print(Offset)
