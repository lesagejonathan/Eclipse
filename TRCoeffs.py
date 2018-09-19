import numpy as np
from numpy.linalg import solve

def ElementDirectivity(th,f,e,c):

    kx = np.sin(np.deg2rad(th))*f*np.pi*2./c

    D = np.abs(np.sinc(0.5*e*kx/(np.pi)))

    if len(D)>1:

        return D/np.amax(D)

def GratingLobeCheck(th,f,p,c,fracbnd=0.5):

    f = np.array([f-(fracbnd/2)*f,f+(fracbnd/2)*f])

    lmbd = c/f

    warn = np.all(lmbd/(1+np.abs(np.sin(np.deg2rad(th))))>p)

    return warn


def GetAngle(th,c):

    return np.arcsin(np.sin(th)*(c[1]/c[0])+0j)


def ComputeTRCoeffs(IncidentMode, IncidentAmplitude, Angle, cL, cT, rho):

    """ Output is array with [RL, RT, TL, TT, ERL, ERT, ETL, ETT]

    where,

    RL - Longitudinal displacement amplitude reflection coefficient,
    RT - Shear displacement amplitude reflection coefficient,
    TL - Longitudinal displacement amplitude transmission coefficient,
    TT - Shear displacement amplitude transmission coefficient,
    ERL - Longitudinal energy reflection coefficient,
    ERT - Shear energy reflection coefficient,
    ETL - Longitudinal energy transmission coefficient,
    TT - Shear energy transmission coefficient,
    

    """

    cL1 = cL[0]
    cL2 = cL[1]
    cT1 = cT[0]
    cT2 = cT[1]
    rho1 = rho[0]
    rho2 = rho[1]



    ZL1 = cL1*rho1

    ZL2 = cL2*rho2

    ZT1 = cT1*rho1

    ZT2 = cT2*rho2

    a1 = cL1/cT1
    b1 = (cT2*cL1*ZT2)/(cL2*cT1*ZT1)
    c1 = (cL1*ZT2)/(cT1*ZT1)
    a2 = cT1/cL1
    b2 = ZL2/ZL1
    c2 = ZT2/ZL1
    b3 = ZT2/ZT1
    c3 = (cT2*ZT2)/(cL2*ZT1)
    c4 = ZL2/ZT1

    thi = np.deg2rad(Angle)

    # mode = IncidentMode['Mode']
    #
    # A = IncidentMode['Amplitude']

    A = np.zeros((4,4),dtype=np.complex)


    if IncidentMode=='Longitudinal':

        thL1 = thi

        thT1 = GetAngle(thi,(cL1,cT1))

        thL2 = GetAngle(thi,(cL1,cL2))

        thT2 = GetAngle(thi,(cL1,cT2))

        # Reference 1

        A[0,0] = np.sin(thi)
        A[0,1] = np.cos(thT1)
        A[0,2] = -np.sin(thL2)
        A[0,3] = np.cos(thT2)

        A[1,0] = np.cos(thi)
        A[1,1] = -np.sin(thT1)
        A[1,2] = np.cos(thL2)
        A[1,3] = np.sin(thT2)

        A[2,0] = np.sin(2*thi)
        A[2,1] = a1*np.cos(2*thT1)
        A[2,2] = b1*np.sin(2*thL2)
        A[2,3] = - c1*np.cos(2*thT2)

        A[3,0] = np.cos(2*thT1)
        A[3,1] = -a2*np.sin(2*thT1)
        A[3,2] = -b2*np.cos(2*thT2)
        A[3,3] = -c2*np.sin(2*thT2)


        B = np.zeros(4,dtype=np.complex)



        B[0] = - np.sin(thi)
        B[1] = np.cos(thi)
        B[2] = np.sin(2*thi)
        B[3] = -np.cos(2*thT1)


        B = B.reshape(-1,1)

        RT = solve(A,IncidentAmplitude*B).flatten()


        ERT = np.zeros((4,1),dtype=np.complex)

        ERT[0] = np.abs(RT[0])**2

        ERT[1] = (np.abs(RT[1])**2)*(cT1*np.cos(thT1))/(cL1*np.cos(thi))

        ERT[2] =(np.abs(RT[2]**2)*(rho2*cL2*np.cos(thL2))/(rho1*cL1*np.cos(thi)))

        ERT[3] = (np.abs(RT[3]**2)*(rho2*cT2*np.cos(thT2))/(rho1*cL1*np.cos(thi)))


    elif IncidentMode=='Shear':

        thT1 = thi

        thL1 = GetAngle(thi,(cT1,cL1))

        thL2 = GetAngle(thi,(cT1,cL2))

        thT2 = GetAngle(thi,(cT1,cT2))


        A[0,0] = np.sin(thL1)
        A[0,1] = np.cos(thi)
        A[0,2] = -np.sin(thL2)
        A[0,3] = np.cos(thT2)

        A[1,0] = np.cos(thL1)
        A[1,1] = -np.sin(thi)
        A[1,2] = np.cos(thL2)
        A[1,3] = np.sin(thT2)

        A[2,0] = a2*np.sin(2*thL1)
        A[2,1] = np.cos(2*thi)
        A[2,2] = c3*np.sin(2*thL2)
        A[2,3] = - b3*np.cos(2*thT2)

        A[3,0] = a1*np.cos(2*thL1)
        A[3,1] = -np.sin(2*thi)
        A[3,2] = -c4*np.cos(2*thT2)
        A[3,3] = -b3*np.sin(2*thT2)

        B = np.zeros((4,1),dtype=np.complex)

        B[0] = np.cos(thi)
        B[1] = np.sin(thi)
        B[2] = -np.cos(2*thi)
        B[3] = -np.sin(2*thi)


        RT = solve(A,IncidentAmplitude*B).flatten()

        ERT = np.zeros((4,1))

        ERT[0] = (np.abs(RT[0])**2)*(cL1*np.cos(thL1))/(cT1*np.cos(thi))

        ERT[1] = (np.abs(RT[1]))**2

        ERT[3] = (np.abs(RT[3])**2)*(rho2*cT2*np.cos(thT2))/(rho1*cT1*np.cos(thi))

        ERT[2] = (np.abs(RT[2])**2)*(rho2*cL2*np.cos(thL2))/(rho1*cT2*np.cos(thi))


    if np.abs(np.imag(thL2))>0:

        RT[2] = 0.

        ERT[2] = 0.


    if np.abs(np.imag(thT2))>0:

        RT[3] = 0.

        ERT[3] = 0.

    return np.real(np.concatenate((RT.flatten(),ERT.flatten()))).flatten()
