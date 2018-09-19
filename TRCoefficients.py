<<<<<<< HEAD
import numpy as np


def ComputeTRCoeffs(IncidentMode,MaterialProperties):


    cL1 = MaterialProperties['LongitudinalVelocity'][0]

    cL2 = MaterialProperties['LongitudinalVelocity'][1]

    cT1 = MaterialProperties['ShearVelocity'][0]

    cT2 = MaterialProperties['ShearVelocity'][1]

    rho1 = MaterialProperties['Density'][0]

    rho2 = MaterialProperties['Density'][1]

    thi = np.deg2rad(IncidentMode['Angle'])


    mode = IncidentMode['Mode']






    a = np.array([])
=======
from numpy import *
from numpy.fft import *
from matplotlib.pylab import *
from scipy.ndimage import *
import os
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
import pickle
from functools import reduce
from Signal import *

TRDict={}
TRDict['WedgeToSteel'] = lambda ti,ttl,tts,cw,css,csL,rs,rw: (array([[cos(ti)/cw, cos(ttl)/csL, -sin(tts)/css], [rw*(0.5*sin(ti)**2 - 1.0), 1.0*rs - 2.0*css**2*rs*sin(ttl)**2/csL**2, -1.0*rs*sin(2*tts)], [0, css**2*rs*sin(2*ttl)/csL**2, rs*cos(2*tts)]]),array([[cos(ti)/cw], [-rw*(0.5*sin(ti)**2 - 1.0)], [0]]))
TRDict['SteelToWedgeCompressionIncidence'] = lambda ti,ttl,trs,trl,cw,css,csL,rs,rw: (array([[cos(ttl)/cw, cos(trl)/csL, sin(trs)/css], [rw*(0.5*sin(ttl)**2 - 1.0), 1.0*rs - 2.0*css**2*rs*sin(trl)**2/csL**2, 1.0*rs*sin(2*trs)], [0, -css**2*rs*sin(2*trl)/csL**2, rs*cos(2*trs)]]),array([[cos(ti)/csL], [-1.0*rs + 2.0*css**2*rs*sin(ti)**2/csL**2], [-css**2*rs*sin(2*ti)/csL**2]]))
TRDict['SteelToWedgeShearIncidence'] = lambda ti,ttl,trs,trl,cw,css,csL,rs,rw: (array([[cos(ttl)/cw, cos(trl)/csL, sin(trs)/css], [rw*(0.5*sin(ttl)**2 - 1.0), 1.0*rs - 2.0*css**2*rs*sin(trl)**2/csL**2, 1.0*rs*sin(2*trs)], [0, -css**2*rs*sin(2*trl)/csL**2, rs*cos(2*trs)]]),array([[-sin(ti)/css], [1.0*rs*sin(2*ti)], [-rs*cos(2*ti)]]))
TRDict['SteelToVacuumShearIncidence'] = lambda ti,trs,trl,css,csL,rs: (array([[-rs + 2*css**2*rs*sin(trl)**2/csL**2, -rs*sin(2*trs)], [-css**2*rs*sin(2*trl)/csL**2,  rs*cos(2*trs)]]),array([[-rs*sin(2*ti)],[-rs*cos(2*ti)]]))
TRDict['SteelToVacuumCompressionIncidence'] = lambda ti,trs,trl,css,csL,rs: (array([[-rs + 2*css**2*rs*sin(trl)**2/csL**2, -rs*sin(2*trs)],[-css**2*rs*sin(2*trl)/csL**2,  rs*cos(2*trs)]]),array([[rs - 2*css**2*rs*sin(ti)**2/csL**2],[-css**2*rs*sin(2*ti)/csL**2]]))


css = 3.24
csL= 5.9
cw = 2.33
phicl = arcsin(cw/csL)*180/pi
phics = arcsin(cw/css)*180/pi
rs = 7.8
rw = 1.05


# Wedge to Steel
ti = arange(1,89)*pi/180
ttl = []
tts = []
for i in range(len(ti)):
    ttl.append(arcsin(complex((csL/cw)*sin(ti[i]))))
    tts.append(arcsin(complex((css/cw)*sin(ti[i]))))

RL = []
RS = []
TL = []
TS = []

for i in range(len(ti)):

    M = TRDict['WedgeToSteel'](ti[i],ttl[i],tts[i],cw,css,csL,rs,rw)

    rl,tl,ts = solve(M[0],M[1])

    RL.append(rl)
    TL.append(tl)
    TS.append(ts)



    # Steel to Vaccum
    # ti = arange(1,89)*pi/180
    # ttl = []
    # tts = []
    # for i in range(len(ti)):
    #     ttl.append(arcsin(complex((csL/cw)*sin(ti[i]))))
    #     tts.append(arcsin(complex((css/cw)*sin(ti[i]))))
    #
    # RL = []
    # RS = []
    # TL = []
    # TS = []
    #
    # # trl = ti
    # # trs=[]
    # # for i in range(len(ti)):
    # #     trs.append(arcsin(complex((css/csL)*sin(ti[i]))))
    #
    #
    # for i in range(len(ti)):
    #
    #     M = TRDict['WedgeToSteel'](ti[i],ttl[i],tts[i],cw,css,csL,rs,rw)
    #
    #     rl,tl,ts = solve(M[0],M[1])
    #
    #     # M = TRDict['SteelToVacuumCompressionIncidence'](ti[i],trs[i],trl[i],css,csL,rs)
    #     # rl,rrs = solve(M[0],M[1])
    #
    #     RL.append(rl)
    #     # RS.append(rrs)
    #     TL.append(tl)
    #     TS.append(ts)
>>>>>>> 80be20956c9af075ba564119461fb1e956b2f2e6
