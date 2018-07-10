import FMC as f
import numpy as np
import _pickle as pickle
from scipy.signal import hilbert
import itertools
import matplotlib.pylab as plt

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

    return np.array([AScans[n,n,:] for n in range(AScans.shape[0])])

def EstimateWedgeParameters(AScans,fs,h,cw,angle,p):

    from scipy.signal import hilbert

    A = KeepDiagonalElements(AScans)

    B = np.zeros(A.shape[0]-3)

    for i in range(A.shape[0]-3):

        eh = h + i*p*np.sin(angle*np.pi/180)

        t = int(np.round(fs*2*eh/cw))

        w = int(np.round(fs*0.25*eh/cw))

        # print(str(i))
        # print(str(t))
        # print(str(w))

        b = np.argmax(np.abs(hilbert(A[i,t-w:t+w]))) + t - w

        B[i] = (0.5*cw*b)/fs

    Co = np.polyfit(np.array(range(A.shape[0]-3)),B,1)

    # print(Co[0])
    # print(Co[1])
    #
    # plt.plot(B)
    # plt.show()



    # return {'Height':Co[1], 'Angle':np.arcsin(C0[0]/p)*180/np.pi}
    return {'Height':Co[1], 'Angle':np.arcsin(Co[0]/p)*180/np.pi}


fs, cw, T, cs, cl, p, h, angle, N = 25., 1.48, 9.5, 3.24, 5.9, 0.6, 20., 21., 32

Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/Husky/HuskyScan2.p','rb'))

WedgeParameter = {'Height':h ,'Velocity': cw, 'Angle':angle}



F = f.LinearCapture(fs,[Scans['AScans'][0]],p,64,WedgeParameters=WedgeParameter)

F.KeepElements(range(N))

A = KeepDiagonalElements(F.AScans[0])
#
# WP = EstimateWedgeParameters(Scans['AScans'][0],fs,h,cw,angle,p)

WP = EstimateWedgeParameters(F.AScans[0],fs,h,cw,angle,p)


WedgeParameter = {'Height':WP['Height'],'Velocity': cw, 'Angle':WP['Angle']}

h = WedgeParameter['Height']
angle = WedgeParameter['Angle'] *np.pi/180
cw = WedgeParameter['Velocity']


B = F.PlaneWaveSweep(0,(np.array([-angle]),np.array([-angle])),(range(N),range(N)),cw)

# Th = MeasureThickness(F.AScans[0],fs,N,p,h,angle,cw,cs,T)
#
# F = F.LinearCapture(fs,Scans[0],p,N,WedgeParameters=WedgeParameter)
#
# del(F)
#
# AScan = F.PlaneWaveSweep(0, 0, (np.array(range(32)),np.array(range(32,64))), cw)
#
# Offset = MeasureProbeOffset(AScan,fs,N,p,h,angle,cw,cs,Th)
#
# WeldCap, Haz = 16, 2
#
# xres = 0.1
# yres = 0.1
#
# xstart = -0.5 * (WeldCap + Haz)
# xend = -xstart
# ystart = 0
# yend = -Th
#
# F.SetRectangularGrid(xstart,xend,ystart,yend,xres,yres)
# #
# # HuskyDelays = {}
# #
# # c = list(itertools.product([cs,cl],repeat=2))
# # HuskyDelays['DirectDirect'] = [(c[i], F.GetDelays['DirectDirect'](c[i],Offset)) for i in range(len(c))]
# #
# # c = list(itertools.product([cs,cl],repeat=3))
# # HuskyDelays['BackWallDirect'] = [(c[i], F.GetDelays['BackwallDirect'](c[i],Th,Offset)) for i in range(len(c))]
# #
# # c = list(itertools.product([cs,cl],repeat=4))
# # HuskyDelays['BackwallBackwall'] = [(c[i], F.GetDelays['BackwallBackwall'](c[i],Th,Offset)) for i in range(len(c))]
