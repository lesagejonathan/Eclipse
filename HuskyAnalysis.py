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

    Co = np.polyfit(np.array(range(A.shape[0])),B,1)

    return {'Height':Co[1], 'Angle':np.arcsin(Co[0]/p)*180/np.pi}

# For Husky:
# fs, cw, T, cs, cl, p, h, angle, N = 25., 1.48, 25.4, 3.24, 5.9, 0.6, 32, 13, 32

# Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/LCalibration.p','rb'))

# For SegWedge:

# fs, cw, T, cs, cl, p, h, angle, N = 25., 2.33, 9.5, 3.24, 5.9, 0.6, 6.7, 31.9, 32
#
# Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/BackwallTest.p','rb'))

# For NotchPlate:

fs, cw, T, cs, cl, p, h, angle, N = 25., 2.33, 20, 3.24, 5.9, 0.6, 5.11, 39, 16

# Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/NotchPlatePitchCatch.p','rb'))

Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/Probe1nw.p','rb'))

WedgeParameter = {'Height':h ,'Velocity': cw, 'Angle':angle}

F = f.LinearCapture(fs,[Scans['AScans'][0]],p,16,WedgeParameters=WedgeParameter)

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

fs, cw, T, cs, cl, p, h, angle, N = 25., 2.33, 20, 3.24, 5.9, 0.6, 5.11, 39, 16

Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/Probe1nw.p','rb'))

WedgeParameter = {'Height':h ,'Velocity': cw, 'Angle':angle}

F = f.LinearCapture(fs,[Scans['AScans'][1]],p,16,WedgeParameters=WedgeParameter)

F.ProcessScans(100,100)

# F.KeepElements(range(16,32))

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

#
#
#
#
# xres = 0.1
# yres = 0.1
#
# xstart = 5
# xend = 30
# ystart = 2
# yend = Th-yres
#
# Offset = 0
#
# # del(F)
# #
# # F = f.LinearCapture(fs,[Scans['AScans'][1]],p,N,WedgeParameters=WedgeParameter)
# #
# # F.SetRectangularGrid(xstart,xend,ystart,yend,xres,yres)
# #
# # SegWedgeDelays = {}
# #
# # SegWedgeDelays['SDirect'] = F.GetDelays('Direct',cs,Offset)
# # print('First Done')
# # SegWedgeDelays['SBackwall'] = F.GetDelays('Backwall',(cs,cs),Offset,Th)
# # print('Second Done')
# #
# # pickle.dump(SegWedgeDelays,open('/mnt/c/Users/mmarvasti/Desktop/MoScans/SegmentWedgeDelays.p','wb'))
#
#
# Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/BackwallTest.p','rb'))
#
# SegWedgeDelays = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/SegmentWedgeDelays.p','rb'))
#
# del(F)
#
# F = f.LinearCapture(fs,[Scans['AScans'][0]],p,N,WedgeParameters=WedgeParameter)
#
# F.SetRectangularGrid(xstart,xend,ystart,yend,xres,yres)
#
# F.ProcessScans(100,100)
#
# F.Delays = (SegWedgeDelays['SDirect'][0],SegWedgeDelays['SDirect'][1])
#
# I = F.ApplyTFM(0)
#
# del(F)
#
# F = f.LinearCapture(fs,[Scans['AScans'][0]],p,N,WedgeParameters=WedgeParameter)
#
# F.SetRectangularGrid(xstart,xend,ystart,yend,xres,yres)
#
# F.ProcessScans(100,100)
#
# F.Delays = (SegWedgeDelays['SBackwall'][0],SegWedgeDelays['SDirect'][0])
#
# g = F.ApplyTFM(0)






# del(F)
#
# F = f.LinearCapture(fs,[Scans['AScans'][1]],p,N,WedgeParameters=WedgeParameter)
#
# F.ProcessScans(100,100)
#
# F.SetRectangularGrid(xstart,xend,ystart,yend,xres,yres)
#
# F.Delays = SegWedgeDelays['SDirect']
#
# print(len(F.Delays[1]))
# print(F.Delays[1][0].shape)
#
# HuskyTFM = {}
#
# HuskyTFM['PCLflip1'] = [F.ApplyTFM(i,Elements,PitchCatch=True) for i in range(len(Scans['AScans']))]
#
# print('PCLflip1 Done')



# HuskyDelays2['SBackwall'] = F.GetDelays('Backwall',(cs,cs),Offset,Th,PitchCatch=True)
# print('Second Done')



# del (F)
#
# F = f.LinearCapture(fs,[Scans['AScans'][0]],p,64,WedgeParameters=WedgeParameter)
#
# F.ProcessScans(100,100)
#
# F.KeepElements(range(32,64))
#
# WP = EstimateWedgeParameters(F.AScans[0],fs,h,cw,angle,p)
#
# WedgeParameter = {'Height':WP['Height'],'Velocity': cw, 'Angle':WP['Angle']}
#
# h = WedgeParameter['Height']
# angle = WedgeParameter['Angle'] *np.pi/180
# cw = WedgeParameter['Velocity']
#
# B = F.PlaneWaveSweep(0,(np.array([-angle]),np.array([-angle])),(range(N),range(N)),cw)
#
# Th = MeasureThickness(B,fs,N,p,h,angle,cw,cl,T)
#
# print('Next')
# print(h)
# print(angle*180/np.pi)
# print(Th)
#
#
#
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
# del(F)
#
# Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/Sample10Compression.p','rb'))
#
# F = f.LinearCapture(fs,[Scans['AScans'][2]],p,64,WedgeParameters=WedgeParameter)
#
# F.ProcessScans(100,100)
#
# F.KeepElements(range(N))
#
# B = F.PlaneWaveSweep(0,(np.array([-angle]),np.array([-angle])),(range(N),range(N)),cw)
#
# T = 10
#
# Th = MeasureThickness(B,fs,N,p,h,angle,cw,cl,T)
#
# # print(Th)
#
#
# WeldCap, Haz = 16, 2
#
# xres = 0.1
# yres = 0.1
#
# xstart = -0.5 * (WeldCap + Haz)
# xend = -xstart
# ystart = 0
# yend = 11
# #
#
# F.SetRectangularGrid(xstart,xend,ystart,yend,xres,yres)
#
# F.NumberOfElements = N
#
# HuskyDelays2 = {}
#
# HuskyDelays2['LBackwall'] = F.GetDelays('Backwall',(cl,cl),Offset,Th,PitchCatch=True)
# print('First Done')
# HuskyDelays2['SBackwall'] = F.GetDelays('Backwall',(cs,cs),Offset,Th,PitchCatch=True)
# print('Second Done')
# # HuskyDelays['LDirect'] = F.GetDelays('Direct',cl,Offset,PitchCatch=True)
# # print('Third Done')
# # HuskyDelays['SDirect'] = F.GetDelays('Direct',cs,Offset,PitchCatch=True)
# # print('Forth Done')
# #
# pickle.dump(HuskyDelays2,open('/mnt/c/Users/mmarvasti/Desktop/MoScans/HuskyDelays2.p','wb'))


# Rendering TFM

# del(F)
#
# delays = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/HuskyDelays.p','rb'))
#
# Scans = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoScans/Sample10Compression.p','rb'))
# #
# # # F = f.LinearCapture(fs,[Scans['AScans'][5]],p,64,WedgeParameters=WedgeParameter)
# #
# F = f.LinearCapture(fs,Scans['AScans'],p,64,WedgeParameters=WedgeParameter)
# #
# # # F.KeepElements(np.array(range(32)))
# #
# F.ProcessScans(100,100)
#
# F.SetRectangularGrid(xstart,xend,ystart,yend,xres,yres)
#
# Elements = (range(32), range(32,64))
#
#
# F.Delays = (delays['LDirect'][0],list(np.flip(delays['LDirect'][0],axis=1)))
#
# print(len(F.Delays[1]))
# print(F.Delays[1][0].shape)
#
# HuskyTFM = {}
#
# HuskyTFM['PCLflip1'] = [F.ApplyTFM(i,Elements,PitchCatch=True) for i in range(len(Scans['AScans']))]
#
# print('PCLflip1 Done')

# F.Delays = (np.array(delays['LDirect'][0]),np.array(np.flip(delays['LDirect'][0],axis=0)))
#
# HuskyTFM['PCLflip0'] = [F.ApplyTFM(i,Elements,PitchCatch=True) for i in range(len(Scans['AScans']))]
#
# print('PCLflip0 Done')
#
# F.Delays = (np.array(delays['LDirect'][0]),np.array(delays['LDirect'][0]))
#
# HuskyTFM['PCLNoflip'] = [F.ApplyTFM(i,Elements,PitchCatch=True) for i in range(len(Scans['AScans']))]
#
# print('Noflip Done')

#





# F.Delays = (np.array(delays['SDirect'][0]),np.array(np.flip(delays['SDirect'][0],axis=1)))
#
# HuskyTF['PCS'] = [F.ApplyTFM(i,Elements,PitchCatch=True) for i in range(len(Scans['AScans']))]
#
# print('PCS Done')
#
# F.KeepElements(np.array(range(32)))
#
# F.Delays = (np.array(delays['LDirect'][0]),np.array(delays['LDirect'][0]))
#
# HuskyTF['PEL'] = [F.ApplyTFM(i,Elements,PitchCatch=True) for i in range(len(Scans['AScans']))]
#
# print('PEL Done')
#
# F.Delays = (np.array(delays['LDirect'][0]),np.array(delays['LDirect'][0]))
#
# HuskyTF['PES'] = [F.ApplyTFM(i,Elements,PitchCatch=True) for i in range(len(Scans['AScans']))]
#
# print('PES Done')

# F.Delays = (delays['LBackwall'][0],delays['LDirect'][0])



# F.NumberOfElements = 32

# I = F.ApplyTFM(0,(np.array(range(32)),np.array(range(32))))


# Elements = (np.array(range(32)),np.array(range(32,64)))

# Elements = (np.array(range(32)),np.array(range(32)))
# Elements = (np.array(range(32)), np.array(range(32,64)))
#
#
#
# I = [F.ApplyTFM(i,Elements,PitchCatch=True) for i in range(len(Scans['AScans']))]

# I = F.ApplyTFM(0,Elements,PitchCatch=True)

# I = F.ApplyTFM(0)



#
# Delay Caluclation for different Modes:


# c = list(itertools.product([cs,cl],repeat=2))
 # HuskyDelays['DirectDirect'] = [(c[i], F.GetDelays['DirectDirect'](c[i],Offset)) for i in range(len(c))]
#
# c = list(itertools.product([cs,cl],repeat=3))
# HuskyDelays['BackWallDirect'] = [(c[i], F.GetDelays['BackwallDirect'](c[i],Th,Offset)) for i in range(len(c))]
#
# c = list(itertools.product([cs,cl],repeat=4))
# HuskyDelays['BackwallBackwall'] = [(c[i], F.GetDelays['BackwallBackwall'](c[i],Th,Offset)) for i in range(len(c))]
