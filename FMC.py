from functools import reduce
import numpy as np
from numpy.fft import rfft, ifft, fftn, ifftn, fftshift
import os
import multiprocessing
import copy

def NumericalAperture(x,y,L):

    x,y = np.meshgrid(x,y)

    rp = np.sqrt((x - L/2)**2 + y**2)

    rm = np.sqrt((x + L/2)**2 + y**2)

    A = np.sin(0.5*np.arccos((rp**2 + rm**2 - L**2)/(2*rp*rm)))

    A[0,:] = A[1,:]

    return A

def NextPow2(x):
    return int(2**int(np.ceil(np.log2(x))))

def RBF(x, ai, xi, beta):

    return ai * np.exp(-beta * (x - xi)**2)

def FitRBF(x, f, beta):

    from numpy.linalg import solve

    return solve(np.exp(-beta * (x.reshape(-1, 1) -
                                 x.reshape(1, -1))**2), f.reshape(-1, 1))

def EstimateProbeDelays(Scans, fsamp, p, h, hfraction=0.1, c=5.92):

    from scipy.signal import hilbert

    M = Scans.shape[0]
    N = Scans.shape[1]

    x = np.abs(hilbert(Scans, axis=2))

    W = int(np.round(fsamp * hfraction * h / c))

    Delays = np.zeros((M, N))

    A = np.zeros(M)

    for m in range(M):

        A[m] = np.sqrt(np.max(np.abs(x[m, m, W::])))

        for n in range(N):

            T = int(
                np.round(fsamp * (2 * np.sqrt((0.5 * (n - m) * p)**2 + h**2) / c)))

            indmax = np.argmax(np.abs(x[m, n, T - W:T + W])) + T - W - T

            Delays[m, n] = indmax / fsamp

    return Delays, A.reshape(-1, 1) * A.reshape(1, -1)

def MeasureProbeOffset(AScans,SweepAngle,fs,N,p,h,angle,cw,cs,Th):

    from scipy.signal import hilbert

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

def RearrangeHalfMatrixData(AScans,N):

    d = AScans[:,:,0]
    M = len(d[0,:])
    A = np.zeros((N,N,M))
    m = 0

    for i in range(N):
        for j in range(i,N):
            A[i,j,:] = d[m,:]
            m = m + 1

    return A

class LinearCapture:

    def __init__(self, fs, scans, p, N, probedelays=None, WedgeParameters=None):

        import copy

        self.SamplingFrequency = fs
        self.Pitch = p
        self.NumberOfElements = N

        if probedelays is None:

            self.ProbeDelays = np.zeros((N, N))

        else:

            self.ProbeDelays = probedelays

        self.AScans = copy.deepcopy(scans)

        self.AScans = [a.astype(np.float)/32768. for a in self.AScans]

        self.AmplitudeCorrection = None

        self.WedgeParameters = WedgeParameters

    def EstimateThickness(self, c, Elements, Angle, ScanIndex=None):
    # def EstimateThickness(self,c,Th,Elements, ScanIndex=None):

        from scipy.signal import hilbert
        from scipy.signal import argrelextrema

        p = self.Pitch
        h = self.WedgeParameters['Height']
        cw = self.WedgeParameters['Velocity']
        phii = self.WedgeParameters['Angle']
        phi = phii*np.pi/180
        fs = self.SamplingFrequency
        N = len(Elements)

        if ScanIndex is None:
            ScanIndex = 0

        # h = h + 0.5*p*(N-1)*np.sin(phi)
        #
        # I = int(np.round(fs*2*(h/cw)))

        # A = self.PlaneWaveSweep(ScanIndex, [-phii], (Elements,Elements), cw)

        A = self.PlaneWaveSweep(ScanIndex, [-Angle], (Elements,Elements), cw)

        I = np.argmax(np.abs(hilbert(np.real(A[0,:]))))

        ind = argrelextrema(np.abs(hilbert(np.real(A[0,I+25::]))),np.greater,order=40)

        # T = int(np.round(fs*2*(h/cw + Th/c)))

        # W = int(np.round(0.1*T))

        # b = np.argmax(np.abs(hilbert(np.real(A[0,T-W:T+W])))) + T - W
        #
        # t1 = b/fs
        #
        # TT = int(np.round(fs*2*(h/cw + 2*Th/c)))
        #
        # # W = int(np.round(0.1*T))
        #
        # # b = np.argmax(np.abs(hilbert(np.real(A[0,T-W:T+W])))) + T - W
        # b = np.argmax(np.abs(hilbert(np.real(A[0,TT-W:TT+W])))) + TT - W
        #
        # t2 = b/fs
        #
        # return (0.5*c*(t2-t1),T,TT,N,h)

        return 0.5*c*((ind[0][1]-ind[0][0])/fs)

    def EstimateSweepBasedWedgeAngle(self,Elements,ScanIndex=None):

        N = len(Elements)
        p = self.Pitch
        h = self.WedgeParameters['Height']
        cw = self.WedgeParameters['Velocity']
        phi = self.WedgeParameters['Angle']*np.pi/180
        a = self.WedgeParameters['Angle']
        fs = self.SamplingFrequency

        if ScanIndex is None:
            ScanIndex = 0

        from scipy.signal import hilbert

        AScans = self.AScans[ScanIndex]

        SweepAngles = np.arange(-a-10,-a+10,0.25)

        Elements = (Elements,Elements)

        S = []

        for i in range(len(SweepAngles)):

            A = self.PlaneWaveSweep(ScanIndex,[SweepAngles[i]],Elements,cw)

            # phiw = np.abs(SweepAngles[i]*np.pi/180)
            #
            # T = int(np.round(fs*2*((h + 0.5*(N-1)*p*np.sin(phiw))/cw)))
            #
            # W = int(np.round(0.1*T))
            #
            # S.append(np.amax(np.abs(hilbert(np.real(A[0,T-W:T+W])))))
            S.append(np.amax(np.abs(hilbert(np.real(A[0,:])))))

        return np.abs(SweepAngles[np.argmax(np.array(S))])

    def EstimateWedgeHeight(self,Elements,ScanIndex=None):
    # def EstimateWedgeHeight(self,WedgeAngle,Elements,ScanIndex=None):

        N = self.NumberOfElements
        p = self.Pitch
        h = self.WedgeParameters['Height']
        cw = self.WedgeParameters['Velocity']
        # phi = WedgeAngle*np.pi/180
        fs = self.SamplingFrequency

        if ScanIndex is None:
            ScanIndex = 0

        from scipy.signal import hilbert

        AScans = self.AScans[ScanIndex]

        # A = np.array([AScans[n,n,:] for n in range(AScans.shape[0])])
        A = np.array([AScans[Elements[n],Elements[n],:] for n in range(len(Elements))])

        B = np.zeros(A.shape[0])

        for i in range(A.shape[0]):

            # eh = h + i*p*np.sin(phi)
            #
            # t = int(np.round(fs*2*eh/cw))
            #
            # w = int(np.round(fs*0.25*eh/cw))

            # b = np.argmax(np.abs(hilbert(np.real(A[i,t-w:t+w])))) + t - w

            b = np.argmax(np.abs(hilbert(np.real(A[i,:]))))

            B[i] = (0.5*cw*b)/fs

        Co = np.polyfit(np.array(range(A.shape[0])),B,1)

        return Co[1]

        # return {'Height':Co[1], 'Angle':np.arcsin(Co[0]/p)*180/np.pi}

    def EstimateSweepBasedProbesOffset(self,WedgeAngle,WedgeHeight,c,Th,anglerange,ScanIndex,Elements=None):

        from scipy.signal import hilbert

        N = self.NumberOfElements
        p = self.Pitch
        # h = self.WedgeParameters['Height']
        h = WedgeHeight
        cw = self.WedgeParameters['Velocity']
        # phiw = self.WedgeParameters['Angle']*np.pi/180
        phiw = WedgeAngle*np.pi/180

        fs = self.SamplingFrequency

        if Elements is None:

            Elements = (list(range(int(N/2))),list(range(int(N/2),N)))

        N = len(Elements[0])

        SweepAngles = np.arange(0,int(anglerange),0.2)

        S = []
        # AA = []
        # TT = []

        for i in range(len(SweepAngles)):

            A = self.PlaneWaveSweep(ScanIndex,[SweepAngles[i]],Elements,cw)

            a = SweepAngles[i]*np.pi/180

            phis = np.arcsin((c/cw)*np.sin(phiw+a))

            T = int(np.round(fs*2*(((h + 0.5*(N-1)*p*np.sin(phiw))/(cw*np.cos(phiw+a))) + (Th/(c*np.cos(phis))))))

            W = 5

            S.append(np.amax(np.abs(hilbert(np.real(A[0,T-W:T+W])))))

            # TT.append(T)
            #
            # AA.append(A)

        angle = SweepAngles[np.argmax(np.array(S))]*np.pi/180

        phir = np.arcsin((c/cw)*np.sin(phiw + angle))

        Offset = 2*(0.5*(N-1)*p*np.cos(phiw) + np.tan(phiw+angle)*(h+0.5*(N-1)*p*np.sin(phiw)) + Th*np.tan(phir))

        # return (Offset,AA,TT,np.argmax(np.array(S)),angle)
        return Offset

    def EstimateProbesOffset(self,c,Offset,Thickness,ScanIndex=None):

        from scipy.signal import hilbert
        from scipy.optimize import minimize_scalar,minimize
        from scipy.signal import argrelextrema

        if ScanIndex is None:
            ScanIndex = 0

        p = self.Pitch
        N = self.NumberOfElements
        h = self.WedgeParameters['Height']
        phiw = self.WedgeParameters['Angle']*np.pi/180
        cw = self.WedgeParameters['Velocity']
        Fs = self.SamplingFrequency
        Scans = self.AScans[ScanIndex]
        Th = Thickness
        sphi = np.sin(phiw)
        cphi = np.cos(phiw)
        tphi = np.tan(phiw)
        c0 = c[0]
        c1 = c[1]

        def f(x,n,m,d):

            x0,x1,x2 = x[0],x[1],x[2]

            t = np.sqrt((h + m*p*sphi)**2 + (-cphi*m*p + d - x2)**2)/cw + np.sqrt((h + n*p*sphi)**2 + (cphi*n*p - x0)**2)/cw + np.sqrt(Th**2 + (-x1 + x2)**2)/c1 + np.sqrt(Th**2 + (-x0 + x1)**2)/c0

            dtdx =  [-(cphi*n*p - x0)/(cw*np.sqrt((h + n*p*sphi)**2 + (cphi*n*p - x0)**2)) + (x0 - x1)/(c0*np.sqrt(Th**2 + (x0 - x1)**2)),(x1 - x2)/(c1*np.sqrt(Th**2 + (x1 - x2)**2)) - (x0 - x1)/(c0*np.sqrt(Th**2 + (x0 - x1)**2)),(cphi*m*p - d + x2)/(cw*np.sqrt((h + m*p*sphi)**2 + (cphi*m*p - d + x2)**2)) - (x1 - x2)/(c1*np.sqrt(Th**2 + (x1 - x2)**2))]

            return t, np.array(dtdx)

        def delays(n,m,d):

            xn = tphi*(h+n*p*sphi)+n*p*cphi
            xm = d - (tphi*(h+m*p*sphi)+m*p*cphi)

            xi = (xn,d/2,xm)

            res = minimize(f,xi,args=(n,m,d),method='BFGS',jac='True')

            return res.fun

        def Error(d):

            T = np.array([[int(np.round(Fs*delays(n,m,d))) for m in range(int(N/2))] for n in range(int(N/2))])

            return np.nansum((MeasuredTimes - T)**2)

        T = np.array([[int(np.round(Fs*delays(n,m,Offset))) for m in range(int(N/2))] for n in range(int(N/2))])

        W = np.array([[int(np.round(0.1*T[n,m])) for m in range(int(N/2))] for n in range(int(N/2))])

        MeasuredTimes = np.zeros((int(N/2),int(N/2)))

        for n in range(int(N/2)):

            for m in range(int(N/2)):

                A = np.abs(hilbert(np.real(Scans[n,m,T[n,m]-W[n,m]:T[n,m]+W[n,m]])))

                ind = argrelextrema(A,np.greater,order=10)

                if (len(ind[0])<1):
                    MeasuredTimes[n,m] = np.nan

                elif ((len(ind[0])<2) & (len(ind[0])>0)):

                    MeasuredTimes[n,m] = ind[0] + T[n,m] - W[n,m]

                else:

                    Ind = np.argmin(np.abs(W[n,m] - ind[0]))
                    MeasuredTimes[n,m] = ind[0][Ind] + T[n,m] - W[n,m]

        res = minimize_scalar(Error,(Offset-5.,Offset+5.))

        return res.x

        return (res,T,MeasuredTimes)

    # def EstimateProbesOffset(self,c,Offset,Thickness):
    #
    #     from scipy.signal import hilbert
    #     from scipy.optimize import minimize_scalar,minimize
    #     from scipy.signal import argrelextrema
    #
    #     p = self.Pitch
    #     N = self.NumberOfElements
    #     h = self.WedgeParameters['Height']
    #     phiw = self.WedgeParameters['Angle']*np.pi/180
    #     cw = self.WedgeParameters['Velocity']
    #     Fs = self.SamplingFrequency
    #     Scans = self.AScans[0]
    #     Th = Thickness
    #     sphi = np.sin(phiw)
    #     cphi = np.cos(phiw)
    #     tphi = np.tan(phiw)
    #     c0 = c[0]
    #     c1 = c[1]
    #
    #     def f(x,n,m,d):
    #
    #         x0,x1,x2 = x[0],x[1],x[2]
    #
    #         t = np.sqrt((h + m*p*sphi)**2 + (-cphi*m*p + d - x2)**2)/cw + np.sqrt((h + n*p*sphi)**2 + (cphi*n*p - x0)**2)/cw + np.sqrt(Th**2 + (-x1 + x2)**2)/c1 + np.sqrt(Th**2 + (-x0 + x1)**2)/c0
    #
    #         dtdx =  [-(cphi*n*p - x0)/(cw*np.sqrt((h + n*p*sphi)**2 + (cphi*n*p - x0)**2)) + (x0 - x1)/(c0*np.sqrt(Th**2 + (x0 - x1)**2)),(x1 - x2)/(c1*np.sqrt(Th**2 + (x1 - x2)**2)) - (x0 - x1)/(c0*np.sqrt(Th**2 + (x0 - x1)**2)),(cphi*m*p - d + x2)/(cw*np.sqrt((h + m*p*sphi)**2 + (cphi*m*p - d + x2)**2)) - (x1 - x2)/(c1*np.sqrt(Th**2 + (x1 - x2)**2))]
    #
    #         return t, np.array(dtdx)
    #
    #     def delays(n,m,d):
    #
    #         xn = tphi*(h+n*p*sphi)+n*p*cphi
    #         xm = d - (tphi*(h+m*p*sphi)+m*p*cphi)
    #
    #         xi = (xn,d/2,xm)
    #
    #         res = minimize(f,xi,args=(n,m,d),method='BFGS',jac='True')
    #
    #         return res.fun
    #
    #     def Error(d):
    #
    #         T = np.array([[int(np.round(Fs*delays(n,m,d))) for m in range(int(N/2))] for n in range(int(N/2))])
    #         # T = np.array([[int(np.round(Fs*delays(Elements[0][n],Elements[1][m],d))) for m in range(int(mm))] for n in range(int(mm))])
    #         # T = np.array([[int(np.round(Fs*delays(Elements[0][n],Elements[1][m],d))) for m in range(nn)] for n in range(nn)])
    #
    #         return np.sum(np.sum((MeasuredTimes - T)**2))
    #         # return np.sum(np.sum((MT - T)**2))
    #
    #     T = np.array([[int(np.round(Fs*delays(n,m,Offset))) for m in range(int(N/2))] for n in range(int(N/2))])
    #
    #     W = np.array([[int(np.round(0.15*T[n,m])) for m in range(int(N/2))] for n in range(int(N/2))])
    #
    #     # MeasuredTimes = np.array([[np.argmax(np.abs(hilbert(np.real(Scans[n,m+int(N/2),T[n,m]-W[n,m]:T[n,m]+W[n,m]]))))+T[n,m]-W[n,m] for m in range(int(N/2))] for n in range(int(N/2))])
    #     # MeasuredTimes = np.array([[np.argmax(np.abs(hilbert(np.real(Scans[n,m,T[n,m]-W[n,m]:T[n,m]+W[n,m]]))))+T[n,m]-W[n,m] for m in range(int(N/2))] for n in range(int(N/2))])
    #     MeasuredTimes = np.array([[np.argmax(np.abs(hilbert(np.real(Scans[n,m+int(N/2),T[n,m]-W[n,m]:T[n,m]+W[n,m]]))))+T[n,m]-W[n,m] for m in range(int(N/2))] for n in range(int(N/2))])
    #
    #     # error = np.abs(T-MeasuredTimes)
    #     #
    #     # mm = np.arange(16,32,2)
    #     #
    #     # X = []
    #     #
    #     # for i in range(len(mm)):
    #     #
    #     #     ind = np.argsort(error.flatten())[0:int(mm[i])]
    #     #
    #     #     Elements= np.unravel_index(np.array(ind),error.shape)
    #     #
    #     #     MT = [[MeasuredTimes[Elements[0][n],Elements[1][m]] for m in range(int(mm[i]))] for n in range(int(mm[i]))]
    #     #
    #     #     nn = int(mm[i])
    #     #
    #     #     res = minimize_scalar(Error,(Offset-5.,Offset+5.))
    #     #
    #     #     X.append(res.x)
    #     #
    #     # return X
    #
    #     res = minimize_scalar(Error,(Offset-5.,Offset+5.))
    #
    #     return res

    def SetRectangularGrid(self,xstart,xend,ystart,yend,xres,yres):

        Nx = np.floor((xend - xstart)/xres) + 1
        Ny = np.floor((yend - ystart)/yres) + 1

        x = np.linspace(xstart,xend,Nx)
        y = np.linspace(ystart,yend,Ny)

        self.xRange = x
        self.yRange = y

    def GetWedgeDelays(self, c, offset):

        from scipy.optimize import minimize_scalar,minimize

        p = self.Pitch
        h = self.WedgeParameters['Height']

        cw = self.WedgeParameters['Velocity']

        cphi = np.cos(self.WedgeParameters['Angle'] * np.pi / 180.)
        sphi = np.sin(self.WedgeParameters['Angle'] * np.pi / 180.)

        def f(X,Y,n):

            P = np.zeros(5)

            P[0]=-c**2 + cw**2
            P[1]=2*X*c**2 - 2*X*cw**2 + 2*c**2*cphi*n*p - 2*cphi*cw**2*n*p
            P[2]=-X**2*c**2 + X**2*cw**2 - 4*X*c**2*cphi*n*p + 4*X*cphi*cw**2*n*p - Y**2*c**2 - c**2*cphi**2*n**2*p**2 + cphi**2*cw**2*n**2*p**2 + cw**2*h**2 + 2*cw**2*h*n*p*sphi + cw**2*n**2*p**2*sphi**2
            P[3]=2*X**2*c**2*cphi*n*p - 2*X**2*cphi*cw**2*n*p + 2*X*c**2*cphi**2*n**2*p**2 - 2*X*cphi**2*cw**2*n**2*p**2 - 2*X*cw**2*h**2 - 4*X*cw**2*h*n*p*sphi - 2*X*cw**2*n**2*p**2*sphi**2 + 2*Y**2*c**2*cphi*n*p
            P[4]=-X**2*c**2*cphi**2*n**2*p**2 + X**2*cphi**2*cw**2*n**2*p**2 + X**2*cw**2*h**2 + 2*X**2*cw**2*h*n*p*sphi + X**2*cw**2*n**2*p**2*sphi**2 - Y**2*c**2*cphi**2*n**2*p**2

            r = np.roots(P)

            r = r[(np.real(r)>=0.)&(~(np.abs(np.imag(r))>0.))]


            if len(r)>0:

                x = np.real(r[0])

                return np.sqrt((h + n * p * sphi)**2 + (cphi * n * p - x)**2) / cw + np.sqrt(Y**2 + (X - x)**2) / c

            else:

                return np.nan

        x,y = np.meshgrid(self.xRange + offset,self.yRange)

        ComputeDelays = np.vectorize(f,excluded=['n'])

        delays = [ComputeDelays(x,y,n) for n in range(self.NumberOfElements)]

        self.Delays = (delays,delays)

    def GetWedgeBackwallDelays(self, c, Th, offset):

        from scipy.optimize import minimize_scalar,minimize

        p = self.Pitch
        h = self.WedgeParameters['Height']

        cw = self.WedgeParameters['Velocity']

        cphi = np.cos(self.WedgeParameters['Angle'] * np.pi / 180.)
        sphi = np.sin(self.WedgeParameters['Angle'] * np.pi / 180.)

        c1 = c[0]
        c2 = c[1]

        def f(x,X,Y,n):

            x0,x1 = x[0],x[1]

            t = np.sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x0)**2)/cw + np.sqrt((Th - Y)**2 + (X - x1)**2)/c2 + np.sqrt(Th**2 + (-x0 + x1)**2)/c1

            dtdx = [-(cphi*n*p - x0)/(cw*np.sqrt((h + n*p*sphi)**2 + (cphi*n*p - x0)**2)) + (x0 - x1)/(c1*np.sqrt(Th**2 + (x0 - x1)**2)),-(X - x1)/(c2*np.sqrt((Th - Y)**2 + (X - x1)**2)) - (x0 - x1)/(c1*np.sqrt(Th**2 + (x0 - x1)**2))]

            return t,np.array(dtdx)

        def delays(X,Y,n):

            bnds = ((n*p*cphi,X),(n*p*cphi,X))

            xi = (0.5*(bnds[0][1] + bnds[0][0]),0.5*(bnds[0][1] + bnds[0][0]))

            res = minimize(f,xi,args=(X,Y,n),method='BFGS',jac='True')

            return res.fun

        delays = [[[delays(X,Y,n) for X in self.xRange + offset] for Y in self.yRange] for n in range(self.NumberOfElements)]

        self.Delays = (delays,delays)

def GetWedgeBackwallFrontwallDelays(self, c, Th, Offset):


        from scipy.optimize import minimize_scalar,minimize

        p = self.Pitch
        h = self.WedgeParameters['Height']

        cw = self.WedgeParameters['Velocity']

        cphi = np.cos(self.WedgeParameters['Angle'] * np.pi / 180.)
        sphi = np.sin(self.WedgeParameters['Angle'] * np.pi / 180.)

        c1 = c[0]
        c2 = c[1]
        c3 = c[2]

        def f(x,X,Y,n):

            x0,x1,x2 = x[0],x[1],x[2]

            t = np.sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x0)**2)/cw + np.sqrt(Y**2 + (X - x2)**2)/c3 + np.sqrt(Th**2 + (-x1 + x2)**2)/c2 + np.sqrt(Th**2 + (-x0 + x1)**2)/c1

            dtdx = [-(cphi*n*p - x0)/(cw*np.sqrt((h + n*p*sphi)**2 + (cphi*n*p - x0)**2)) + (x0 - x1)/(c1*np.sqrt(Th**2 + (x0 - x1)**2)),(x1 - x2)/(c2*np.sqrt(Th**2 + (x1 - x2)**2)) - (x0 - x1)/(c1*np.sqrt(Th**2 + (x0 - x1)**2)),-(X - x2)/(c3*np.sqrt(Y**2 + (X - x2)**2)) - (x1 - x2)/(c2*np.sqrt(Th**2 + (x1 - x2)**2))]

            return t,np.array(dtdx)

        def delays(X,Y,n):

            bnds = ((n*p*cphi,X),(n*p*cphi,X),(n*p*cphi,X))

            xi = (0.5*(bnds[0][1] + bnds[0][0]),0.5*(bnds[0][1] + bnds[0][0]),0.5*(bnds[0][1] + bnds[0][0]))

            res = minimize(f,xi,args=(X,Y,n),method='BFGS',jac='True')

            return res.fun

        delays = [[[delays(X,Y,n) for X in self.xRange + Offset] for Y in self.yRange] for n in range(self.NumberOfElements)]

        self.Delays = (delays,delays)

    def SetPitchCatchDelays(self):

        delays = self.Delays

        d = copy.deepcopy(delays[0])

        for i in range(len(d)):

            d[i] = np.flip(d[i],axis=1)

        self.Delays = (delays[0],d)

    def ProcessScans(self, zeropoints=20, bp=10, normalize=True, takehilbert=True):

        from scipy.signal import detrend, hilbert
        from numpy.linalg import norm

        L = self.AScans[0].shape[2]

        d = np.round(self.ProbeDelays*self.SamplingFrequency).astype(int)

        M = self.AScans[0].shape[0]
        N = self.AScans[0].shape[1]

        dmax = np.amax(d)

        Lmax = np.amin(np.abs(L-d))

        if dmax<zeropoints:

            for i in range(len(self.AScans)):

                a = np.zeros((M,N,Lmax))

                for m in range(M):

                    for n in range(N):

                        self.AScans[i][m,n,0:zeropoints] = 0.

                        a[m,n,0:L-d[m,n]] = self.AScans[i][m,n,d[m,n]:L]

                self.AScans[i] = detrend(a, bp=list(np.arange(0, Lmax, bp).astype(int)))

                if takehilbert:

                    self.AScans[i] = hilbert(self.AScans[i])

                if normalize:

                    self.AScans[i] = self.AScans[i]/norm(self.AScans[i])

    def SetGridforPipe(self,Radious,Thickness,Offset,xres,yres,convex = True):

        n = self.NumberOfElements
        p = self.Pitch

        if convex is True:

            ystart = Offset
            yend = Offset + Radious - np.sqrt((Radious - Thickness)**2 - (0.5*p*(n-1))**2)

        else:

            ystart = -(Radious - Thickness - Offset) + np.sqrt((Radious-Thickness)**2 - (0.5*p*(n-1))**2)
            yend = Offset + Thickness

        xstart = -0.5*p*(n-1)
        xend = -xstart

        Nx = np.floor((xend - xstart)/xres) + 1
        Ny = np.floor((yend - ystart)/yres) + 1

        x = np.linspace(xstart,xend,Nx)
        y = np.linspace(ystart,yend,Ny)

        self.xRange = x
        self.yRange = y

    def SetGridforPipe2(self,Radious,Thickness,ArcLength,xres,yres):

        teta = ArcLength/Radious
        xend = Radious*np.sin(teta)
        xstart = -xend

        ystart = (Radious-Thickness)*np.cos(teta)
        yend = Radious

        x = np.arange(xstart,xend,xres)
        y = np.arange(ystart,yend,yres)

        self.xRange = x
        self.yRange = y

    def GetWedgeOnPipeDelays(self,c,Radious,Thickness,FirstElementArcLength):

        from scipy.optimize import minimize_scalar,minimize

        p = self.Pitch
        nn = self.NumberOfElements
        h = self.WedgeParameters['Height']
        a = self.WedgeParameters['Angle']*np.pi/180
        cw = self.WedgeParameters['Velocity']
        R = Radious
        L = FirstElementArcLength
        teta = L/R
        Th = Thickness
        phiw = a + teta - np.arcsin(nn*p*np.cos(a)/(2*(R+h)))
        sphi = np.sin(phiw)
        cphi = np.cos(phiw)

        def f(x,X,Y,n):

            t = np.sqrt((-cphi*n*p + x - (-R - h)*np.sin(teta))**2 + (-n*p*sphi - (R + h)*np.cos(teta) + np.sqrt(R**2 - x**2))**2)/cw + np.sqrt((X - x)**2 + (Y - np.sqrt(R**2 - x**2))**2)/c

            dtdx =  (c*(x*(n*p*sphi + (R + h)*np.cos(teta) - np.sqrt(R**2 - x**2)) + np.sqrt(R**2 - x**2)*(-cphi*n*p + x + (R + h)*np.sin(teta)))*np.sqrt((X - x)**2 + (Y - np.sqrt(R**2 - x**2))**2) + cw*(x*(Y - np.sqrt(R**2 - x**2)) + np.sqrt(R**2 - x**2)*(-X + x))*np.sqrt((-cphi*n*p + x + (R + h)*np.sin(teta))**2 + (n*p*sphi + (R + h)*np.cos(teta) - np.sqrt(R**2 - x**2))**2))/(c*cw*np.sqrt(R**2 - x**2)*np.sqrt((X - x)**2 + (Y - np.sqrt(R**2 - x**2))**2)*np.sqrt((-cphi*n*p + x + (R + h)*np.sin(teta))**2 + (n*p*sphi + (R + h)*np.cos(teta) - np.sqrt(R**2 - x**2))**2))

            return t, dtdx

        def delays(X,Y,n):

            ymin = np.sqrt((R-Th)**2-X**2)
            ymax = np.sqrt(R**2-X**2)

            if (Y>ymin) and (Y<ymax):

                xn = cphi*n*p + (-R - h)*np.sin(teta)

                xi = 0.5*(X + xn)

                res = minimize(f,xi,args=(X,Y,n),method='BFGS',jac='True')

                return res.fun

                # x = res.x
                # y = np.sqrt(R**2-x**2)
                # m = (Y-y)/(X-x)
                #
                # C = np.zeros(3)
                # C[0] = 1+m**2
                # C[1] = -2*m**2*x+2*m*y
                # C[2] = y**2+m**2*x**2-2*m*y*x-(R-Th)**2
                #
                # A = np.roots(C)
                #
                # if (np.isreal(A[0])) or (np.isreal(A[1])):
                #
                #     return np.nan
                #
                # else:
                #
                #     return res.fun

            else:

                return np.nan

        delays = [np.array([[delays(X,Y,n) for X in self.xRange] for Y in self.yRange]) for n in range(self.NumberOfElements)]

        self.Delays = (delays,delays)

    def GetWedgeOnPipeBackwallDelays(self,c,Radious,Thickness,FirstElementArcLength):

        from scipy.optimize import minimize_scalar,minimize

        nn = self.NumberOfElements
        p = self.Pitch
        h = self.WedgeParameters['Height']
        a = self.WedgeParameters['Angle']*np.pi/180
        cw = self.WedgeParameters['Velocity']
        R = Radious
        L = FirstElementArcLength
        teta = L/R
        Th = Thickness
        phiw = a + teta - np.arcsin(nn*p*np.cos(a)/(2*(R+h)))
        sphi = np.sin(phiw)
        cphi = np.cos(phiw)
        c0 = c[0]
        c1 = c[1]

        def f(x,X,Y,n):

            x0,x1 = x[0],x[1]

            t = np.sqrt((-cphi*n*p + x0 - (-R - h)*np.sin(teta))**2 + (-n*p*sphi - (R + h)*np.cos(teta) + np.sqrt(R**2 - x0**2))**2)/cw + np.sqrt((X - x1)**2 + (Y - np.sqrt(-x1**2 + (R - Th)**2))**2)/c1 + np.sqrt((-x0 + x1)**2 + (-np.sqrt(R**2 - x0**2) + np.sqrt(-x1**2 + (R - Th)**2))**2)/c0

            dtdx = [(c0*(x0*(n*p*sphi + (R + h)*np.cos(teta) - np.sqrt(R**2 - x0**2)) + np.sqrt(R**2 - x0**2)*(-cphi*n*p + x0 + (R + h)*np.sin(teta)))*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - Th)**2))**2) + cw*(-x0*(np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - Th)**2)) + np.sqrt(R**2 - x0**2)*(x0 - x1))*np.sqrt((-cphi*n*p + x0 + (R + h)*np.sin(teta))**2 + (n*p*sphi + (R + h)*np.cos(teta) - np.sqrt(R**2 - x0**2))**2))/(c0*cw*np.sqrt(R**2 - x0**2)*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - Th)**2))**2)*np.sqrt((-cphi*n*p + x0 + (R + h)*np.sin(teta))**2 + (n*p*sphi + (R + h)*np.cos(teta) - np.sqrt(R**2 - x0**2))**2)),-X/(c1*np.sqrt((X - x1)**2 + (Y - np.sqrt(-x1**2 + (R - Th)**2))**2)) + Y*x1/(c1*np.sqrt(-x1**2 + (R - Th)**2)*np.sqrt((X - x1)**2 + (Y - np.sqrt(-x1**2 + (R - Th)**2))**2)) - x0/(c0*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - Th)**2))**2)) + x1*np.sqrt(R**2 - x0**2)/(c0*np.sqrt(-x1**2 + (R - Th)**2)*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - Th)**2))**2))]

            return t, np.array(dtdx)

        def delays(X,Y,n):

            xn = cphi*n*p + (-R - h)*np.sin(teta)

            xi = (0.5*(X + xn),0.5*(X + xn))

            res = minimize(f,xi,args=(X,Y,n),method='BFGS',jac='True')

            return res.fun

        delays = [np.array([[delays(X,Y,n) for X in self.xRange] for Y in self.yRange]) for n in range(self.NumberOfElements)]

        self.Delays = (delays,delays)

    def GetWedgeOnPipeBackwallFrontwallDelays(self,c,Radious,Thickness,FirstElementArcLength):

        from scipy.optimize import minimize_scalar,minimize

        p = self.Pitch
        nn = self.NumberOfElements
        h = self.WedgeParameters['Height']
        a = self.WedgeParameters['Angle']*np.pi/180
        cw = self.WedgeParameters['Velocity']
        R = Radious
        L = FirstElementArcLength
        teta = L/R
        Th = Thickness
        phiw = a + teta - np.arcsin(nn*p*np.cos(a)/(2*(R+h)))
        sphi = np.sin(phiw)
        cphi = np.cos(phiw)
        c0 = c[0]
        c1 = c[1]
        c2 = c[2]

        def f(x,X,Y,n):

            x0,x1,x2 = x[0],x[1],x[2]

            t = np.sqrt((cphi*n*p - x0 + (-R - h)*np.sin(teta))**2 + (n*p*sphi + (R + h)*np.cos(teta) - np.sqrt(R**2 - x0**2))**2)/cw + np.sqrt((X - x2)**2 + (Y - np.sqrt(R**2 - x2**2))**2)/c2 + np.sqrt((-x1 + x2)**2 + (np.sqrt(R**2 - x2**2) - np.sqrt(-x1**2 + (R - Th)**2))**2)/c1 + np.sqrt((-x0 + x1)**2 + (-np.sqrt(R**2 - x0**2) + np.sqrt(-x1**2 + (R - Th)**2))**2)/c0

            dtdx = [(c0*(x0*(n*p*sphi + (R + h)*np.cos(teta) - np.sqrt(R**2 - x0**2)) + np.sqrt(R**2 - x0**2)*(-cphi*n*p + x0 + (R + h)*np.sin(teta)))*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - Th)**2))**2) + cw*(-x0*(np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - Th)**2)) + np.sqrt(R**2 - x0**2)*(x0 - x1))*np.sqrt((-cphi*n*p + x0 + (R + h)*np.sin(teta))**2 + (n*p*sphi + (R + h)*np.cos(teta) - np.sqrt(R**2 - x0**2))**2))/(c0*cw*np.sqrt(R**2 - x0**2)*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - Th)**2))**2)*np.sqrt((-cphi*n*p + x0 + (R + h)*np.sin(teta))**2 + (n*p*sphi + (R + h)*np.cos(teta) - np.sqrt(R**2 - x0**2))**2)),x1*np.sqrt(R**2 - x2**2)/(c1*np.sqrt(-x1**2 + (R - Th)**2)*np.sqrt((x1 - x2)**2 + (np.sqrt(R**2 - x2**2) - np.sqrt(-x1**2 + (R - Th)**2))**2)) - x2/(c1*np.sqrt((x1 - x2)**2 + (np.sqrt(R**2 - x2**2) - np.sqrt(-x1**2 + (R - Th)**2))**2)) - x0/(c0*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - Th)**2))**2)) + x1*np.sqrt(R**2 - x0**2)/(c0*np.sqrt(-x1**2 + (R - Th)**2)*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - Th)**2))**2)),-X/(c2*np.sqrt((X - x2)**2 + (Y - np.sqrt(R**2 - x2**2))**2)) + Y*x2/(c2*np.sqrt(R**2 - x2**2)*np.sqrt((X - x2)**2 + (Y - np.sqrt(R**2 - x2**2))**2)) - x1/(c1*np.sqrt((x1 - x2)**2 + (np.sqrt(R**2 - x2**2) - np.sqrt(-x1**2 + (R - Th)**2))**2)) + x2*np.sqrt(-x1**2 + (R - Th)**2)/(c1*np.sqrt(R**2 - x2**2)*np.sqrt((x1 - x2)**2 + (np.sqrt(R**2 - x2**2) - np.sqrt(-x1**2 + (R - Th)**2))**2))]

            return t, np.array(dtdx)

        def delays(X,Y,n):

            ymin = np.sqrt((R-Th)**2-X**2)
            ymax = np.sqrt(R**2-X**2)

            if (Y>ymin) and (Y<ymax):

                xn = cphi*n*p + (-R - h)*np.sin(teta)

                xi = (0.5*(X + xn),0.5*(X + xn),0.5*(X + xn))

                res = minimize(f,xi,args=(X,Y,n),method='BFGS',jac='True')

                return res.fun

                # x = res.x
                # x2 = x[2]
                # y2 = np.sqrt(R**2-x2**2)
                # m = (Y-y2)/(X-x2)
                #
                # C = np.zeros(3)
                # C[0] = 1+m**2
                # C[1] = -2*m**2*x2+2*m*y2
                # C[2] = y2**2 + m**2*x2**2 -2*m*y2*x2 - (R-Th)**2
                #
                # A = np.roots(C)
                #
                # if (np.isreal(A[0])) or (np.isreal(A[1])):
                #
                #     return np.nan
                #
                # else:
                #
                #     return res.fun

            else:

                return np.nan

        delays = [np.array([[delays(X,Y,n) for X in self.xRange] for Y in self.yRange]) for n in range(self.NumberOfElements)]

        self.Delays = (delays,delays)

    def GetCurvedSurfaceDelays(self, Radious, Thickness, Offset, c, Convex = True):

        R = Radious
        h = Thickness
        d = Offset

        from scipy.optimize import minimize_scalar,minimize

        p = self.Pitch
        cw = self.WedgeParameters['Velocity']
        N = self.NumberOfElements

        def f(x,X,Y,n):

            t = np.sqrt((-0.5*p*(-N + 2*n + 1) + x)**2 + (R + d - np.sqrt(R**2 - x**2))**2)/cw + np.sqrt((X - x)**2 + (-R + Y - d + np.sqrt(R**2 - x**2))**2)/c

            dtdx = (c*(x*(R + d - np.sqrt(R**2 - x**2)) + np.sqrt(R**2 - x**2)*(-0.5*p*(-N + 2*n + 1) + x))*np.sqrt((X - x)**2 + (R - Y + d - np.sqrt(R**2 - x**2))**2) + cw*(x*(R - Y + d - np.sqrt(R**2 - x**2)) + np.sqrt(R**2 - x**2)*(-X + x))*np.sqrt((0.5*p*(-N + 2*n + 1) - x)**2 + (R + d - np.sqrt(R**2 - x**2))**2))/(c*cw*np.sqrt(R**2 - x**2)*np.sqrt((X - x)**2 + (R - Y + d - np.sqrt(R**2 - x**2))**2)*np.sqrt((0.5*p*(-N + 2*n + 1) - x)**2 + (R + d - np.sqrt(R**2 - x**2))**2))

            return t, dtdx

        def g(x,X,Y,n):

            t = np.sqrt((0.5*p*(-N + 2*n + 1) - x)**2 + (R - d - h - np.sqrt(x**2 + (R - h)**2))**2)/cw + np.sqrt((X - x)**2 + (R + Y - d - h - np.sqrt(x**2 + (R - h)**2))**2)/c

            dtdx = (-0.5*p*(-N + 2*n + 1) + x - x*(R - d - h - np.sqrt(x**2 + (R - h)**2))/np.sqrt(x**2 + (R - h)**2))/(cw*np.sqrt((0.5*p*(-N + 2*n + 1) - x)**2 + (R - d - h - np.sqrt(x**2 + (R - h)**2))**2)) + (-X + x - x*(R + Y - d - h - np.sqrt(x**2 + (R - h)**2))/np.sqrt(x**2 + (R - h)**2))/(c*np.sqrt((X - x)**2 + (R + Y - d - h - np.sqrt(x**2 + (R - h)**2))**2))

            return t, dtdx

        def delays(X,Y,n):

            print('In Progress')

            if Convex is True:

                ymin = d+R-np.sqrt(R**2-X**2)
                ymax = d+R-np.sqrt((R-h)**2-X**2)

                if (Y>ymin) and (Y<ymax):

                    xi = 0.5*(X + 0.5*p*(1-N+2*n))

                    res = minimize(f,xi,args=(X,Y,n),method='BFGS',jac='True')

                    return res.fun

                    x = res.x
                    y = d + R - np.sqrt(R**2-x**2)
                    m = (Y-y)/(X-x)
                    a = -m*X+Y-d-R

                    C = np.zeros(3)
                    C[0] = 1+m**2
                    C[1] = 2*m*a
                    C[2] = a**2 - (R-h)**2

                    A = np.roots(C)

                    if (np.isreal(A[0])) or (np.isreal(A[1])):

                        return np.nan

                    else:

                        return res.fun

                else:

                    return np.nan

            else:

                ymin = -(R-h-d) + np.sqrt((R-h)**2-X**2)
                ymax = -(R-h-d) + np.sqrt(R**2-X**2)

                if (Y>ymin) and (Y<ymax):

                    xi = 0.5*(X + 0.5*p*(1-N+2*n))

                    res = minimize(g,xi,args=(X,Y,n),method='BFGS',jac='True')

                    return res.fun

                else:

                    return np.nan

        delays = [[[delays(X,Y,n) for X in self.xRange] for Y in self.yRange] for n in range(self.NumberOfElements)]

        self.Delays = (delays,delays)

    def GetCurvedTomographyPaths(self,Radious,Thickness,Offset,c,Convex = True):

        R = Radious
        h = Thickness
        d = Offset
        c1, c2 = c[0], c[1]

        from scipy.optimize import minimize_scalar,minimize

        p = self.Pitch
        cw = self.WedgeParameters['Velocity']
        N = self.NumberOfElements

        def f(x,n,m):

            x0,x1,x2 = x[0],x[1],x[2]

            t = np.sqrt((0.5*p*(-N + 2*m + 1) - x2)**2 + (R + d - np.sqrt(R**2 - x2**2))**2)/cw + np.sqrt((-0.5*p*(-N + 2*n + 1) + x0)**2 + (R + d - np.sqrt(R**2 - x0**2))**2)/cw + np.sqrt((-x1 + x2)**2 + (-np.sqrt(R**2 - x2**2) + np.sqrt(-x1**2 + (R - h)**2))**2)/c2 + np.sqrt((-x0 + x1)**2 + (np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - h)**2))**2)/c1

            dtdx = [(c1*(x0*(R + d - np.sqrt(R**2 - x0**2)) + np.sqrt(R**2 - x0**2)*(-0.5*p*(-N + 2*n + 1) + x0))*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - h)**2))**2) + cw*(-x0*(np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - h)**2)) + np.sqrt(R**2 - x0**2)*(x0 - x1))*np.sqrt((0.5*p*(-N + 2*n + 1) - x0)**2 + (R + d - np.sqrt(R**2 - x0**2))**2))/(c1*cw*np.sqrt(R**2 - x0**2)*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - h)**2))**2)*np.sqrt((0.5*p*(-N + 2*n + 1) - x0)**2 + (R + d - np.sqrt(R**2 - x0**2))**2)),x1*np.sqrt(R**2 - x2**2)/(c2*np.sqrt(-x1**2 + (R - h)**2)*np.sqrt((x1 - x2)**2 + (np.sqrt(R**2 - x2**2) - np.sqrt(-x1**2 + (R - h)**2))**2)) - x2/(c2*np.sqrt((x1 - x2)**2 + (np.sqrt(R**2 - x2**2) - np.sqrt(-x1**2 + (R - h)**2))**2)) - x0/(c1*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - h)**2))**2)) + x1*np.sqrt(R**2 - x0**2)/(c1*np.sqrt(-x1**2 + (R - h)**2)*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x0**2) - np.sqrt(-x1**2 + (R - h)**2))**2)),(c2*(x2*(R + d - np.sqrt(R**2 - x2**2)) + np.sqrt(R**2 - x2**2)*(-0.5*p*(-N + 2*m + 1) + x2))*np.sqrt((x1 - x2)**2 + (np.sqrt(R**2 - x2**2) - np.sqrt(-x1**2 + (R - h)**2))**2) + cw*(-x2*(np.sqrt(R**2 - x2**2) - np.sqrt(-x1**2 + (R - h)**2)) + np.sqrt(R**2 - x2**2)*(-x1 + x2))*np.sqrt((0.5*p*(-N + 2*m + 1) - x2)**2 + (R + d - np.sqrt(R**2 - x2**2))**2))/(c2*cw*np.sqrt(R**2 - x2**2)*np.sqrt((x1 - x2)**2 + (np.sqrt(R**2 - x2**2) - np.sqrt(-x1**2 + (R - h)**2))**2)*np.sqrt((0.5*p*(-N + 2*m + 1) - x2)**2 + (R + d - np.sqrt(R**2 - x2**2))**2))]

            return t, np.array(dtdx)

        def Paths(n,m):

            print('In Progress')

            xi = (0,0,0)

            res = minimize(f,xi,args=(m,n),method='BFGS',jac='True')

            return (res.x, res.fun)

        return [[Paths(n,m) for n in range(N)] for m in range(n)]

    def GetCurvedSurfaceBackwallDelays(self, Radious, Thickness, Offset, c, Convex = True):

        R = Radious
        h = Thickness
        d = Offset
        c0 = c[0]
        c1 = c[1]

        from scipy.optimize import minimize_scalar,minimize

        p = self.Pitch
        cw = self.WedgeParameters['Velocity']
        N = self.NumberOfElements

        def f(x,X,Y,n):

            t = np.sqrt((-0.5*p*(-N + 2*n + 1) + x[0])**2 + (R + d - np.sqrt(R**2 - x[0]**2))**2)/cw + np.sqrt((X - x[1])**2 + (-R + Y - d + np.sqrt(-x[1]**2 + (R - h)**2))**2)/c1 + np.sqrt((-x[0] + x[1])**2 + (np.sqrt(R**2 - x[0]**2) - np.sqrt(-x[1]**2 + (R - h)**2))**2)/c0

            dtdx = [(c0*(x[0]*(R + d - np.sqrt(R**2 - x[0]**2)) + np.sqrt(R**2 - x[0]**2)*(-0.5*p*(-N + 2*n + 1) + x[0]))*np.sqrt((x[0] - x[1])**2 + (np.sqrt(R**2 - x[0]**2) - np.sqrt(-x[1]**2 + (R - h)**2))**2) + cw*(-x[0]*(np.sqrt(R**2 - x[0]**2) - np.sqrt(-x[1]**2 + (R - h)**2)) + np.sqrt(R**2 - x[0]**2)*(x[0] - x[1]))*np.sqrt((0.5*p*(-N + 2*n + 1) - x[0])**2 + (R + d - np.sqrt(R**2 - x[0]**2))**2))/(c0*cw*np.sqrt(R**2 - x[0]**2)*np.sqrt((x[0] - x[1])**2 + (np.sqrt(R**2 - x[0]**2) - np.sqrt(-x[1]**2 + (R - h)**2))**2)*np.sqrt((0.5*p*(-N + 2*n + 1) - x[0])**2 + (R + d - np.sqrt(R**2 - x[0]**2))**2)),(c0*(x[1]*(R - Y + d - np.sqrt(-x[1]**2 + (R - h)**2)) + (-X + x[1])*np.sqrt(-x[1]**2 + (R - h)**2))*np.sqrt((x[0] - x[1])**2 + (np.sqrt(R**2 - x[0]**2) - np.sqrt(-x[1]**2 + (R - h)**2))**2) + c1*(x[1]*(np.sqrt(R**2 - x[0]**2) - np.sqrt(-x[1]**2 + (R - h)**2)) + (-x[0] + x[1])*np.sqrt(-x[1]**2 + (R - h)**2))*np.sqrt((X - x[1])**2 + (R - Y + d - np.sqrt(-x[1]**2 + (R - h)**2))**2))/(c0*c1*np.sqrt(-x[1]**2 + (R - h)**2)*np.sqrt((X - x[1])**2 + (R - Y + d - np.sqrt(-x[1]**2 + (R - h)**2))**2)*np.sqrt((x[0] - x[1])**2 + (np.sqrt(R**2 - x[0]**2) - np.sqrt(-x[1]**2 + (R - h)**2))**2))]

            return t, np.array(dtdx)

        def g(x,X,Y,n):

            x0,x1 = x[0],x[1]

            t = np.sqrt((-0.5*p*(-N + 2*n + 1) + x0)**2 + (-R + d + h + np.sqrt(-x0**2 + (R - h)**2))**2)/cw + np.sqrt((X - x1)**2 + (R + Y - d - h - np.sqrt(R**2 - x1**2))**2)/c1 + np.sqrt((-x0 + x1)**2 + (np.sqrt(R**2 - x1**2) - np.sqrt(-x0**2 + (R - h)**2))**2)/c0

            dtdx = [(c0*(-x0*(-R + d + h + np.sqrt(-x0**2 + (R - h)**2)) + np.sqrt(-x0**2 + (R - h)**2)*(-0.5*p*(-N + 2*n + 1) + x0))*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x1**2) - np.sqrt(-x0**2 + (R - h)**2))**2) + cw*(x0*(np.sqrt(R**2 - x1**2) - np.sqrt(-x0**2 + (R - h)**2)) + (x0 - x1)*np.sqrt(-x0**2 + (R - h)**2))*np.sqrt((0.5*p*(-N + 2*n + 1) - x0)**2 + (-R + d + h + np.sqrt(-x0**2 + (R - h)**2))**2))/(c0*cw*np.sqrt(-x0**2 + (R - h)**2)*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x1**2) - np.sqrt(-x0**2 + (R - h)**2))**2)*np.sqrt((0.5*p*(-N + 2*n + 1) - x0)**2 + (-R + d + h + np.sqrt(-x0**2 + (R - h)**2))**2)),(c0*(-x1*(-R - Y + d + h + np.sqrt(R**2 - x1**2)) + np.sqrt(R**2 - x1**2)*(-X + x1))*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x1**2) - np.sqrt(-x0**2 + (R - h)**2))**2) + c1*(-x1*(np.sqrt(R**2 - x1**2) - np.sqrt(-x0**2 + (R - h)**2)) + np.sqrt(R**2 - x1**2)*(-x0 + x1))*np.sqrt((X - x1)**2 + (-R - Y + d + h + np.sqrt(R**2 - x1**2))**2))/(c0*c1*np.sqrt(R**2 - x1**2)*np.sqrt((X - x1)**2 + (-R - Y + d + h + np.sqrt(R**2 - x1**2))**2)*np.sqrt((x0 - x1)**2 + (np.sqrt(R**2 - x1**2) - np.sqrt(-x0**2 + (R - h)**2))**2))]

            return t, np.array(dtdx)

        def delays(X,Y,n):

            print('In Progress')

            if Convex is True:

                ymin = d+R-np.sqrt(R**2-X**2)
                ymax = d+R-np.sqrt((R-h)**2-X**2)

                if (Y>ymin) and (Y<ymax):

                    xi = 0.5*(X + 0.5*p*(1-N+2*n))

                    xi = (xi,xi)

                    res = minimize(f,xi,args=(X,Y,n),method='BFGS',jac='True')

                    return res.fun

                    x = res.x

                    x0 = x[0]
                    x1 = x[1]

                    y0 = d + R - np.sqrt(R**2 - x0**2)
                    y1 = d + R - np.sqrt((R-h)**2 - x1**2)

                    m = (y1-y0)/(x1-x0)
                    a = -m*x0+y0-d-R

                    C = np.zeros(3)
                    C[0] = 1+m**2
                    C[1] = 2*m*a
                    C[2] = a**2 - (R-h)**2

                    A = np.roots(C)

                    if (np.isreal(A[0])) or (np.isreal(A[1])):

                        return np.nan

                    else:

                        m = (Y-y1)/(X-x1)
                        a = -m*x1+y1-d-R

                        C = np.zeros(3)
                        C[0] = 1+m**2
                        C[1] = 2*m*a
                        C[2] = a**2 - (R-h)**2

                        A = np.roots(C)

                        if (np.isreal(A[0])) or (np.isreal(A[1])):

                            return np.nan

                        else:

                            return res.fun

                else:

                    return np.nan

            else:

                ymin = -(R-h-d) + np.sqrt((R-h)**2-X**2)
                ymax = -(R-h-d) + np.sqrt(R**2-X**2)

                if (Y>ymin) and (Y<ymax):

                    xi = 0.5*(X + 0.5*p*(1-N+2*n))

                    xi = (xi,xi)

                    res = minimize(g,xi,args=(X,Y,n),method='BFGS',jac='True')

                    x = res.x

                    x1 = x[1]
                    y1 = -(R-h-d) + np.sqrt(R**2 - x1**2)

                    m = (Y-y1)/(X-x1)
                    a = -m*x1+y1+R-h-d

                    C = np.zeros(3)
                    C[0] = 1+m**2
                    C[1] = 2*m*a
                    C[2] = a**2 - (R-h)**2

                    A = np.roots(C)

                    if (np.isreal(A[0])) or (np.isreal(A[1])):

                        return np.nan

                    else:

                        return res.fun

                else:

                    return np.nan

        delays = [np.array([[delays(X,Y,n) for X in self.xRange] for Y in self.yRange]) for n in range(self.NumberOfElements)]

        self.Delays = (delays,delays)

    def ReverseElements(self):

        self.AScans = [a[::-1,::-1,:] for a in self.AScans]

    def PlaneWaveSweep(self, ScanIndex, Angles, Elements, c):

        X = np.real(self.AScans[ScanIndex][Elements[0][0]:Elements[0][-1]+1,Elements[1][0]:Elements[1][-1]+1,:])

        L = X.shape[2]

        X = rfft(X, 2*L)

        f = np.linspace(0, self.SamplingFrequency / 2, X.shape[2])

        Ltr = self.Pitch*(len(Elements[0])-1)

        Lrc = self.Pitch*(len(Elements[1])-1)

        dtr = np.linspace(-0.5*Ltr,0.5*Ltr,len(Elements[0]))

        drc = np.linspace(-0.5*Lrc,0.5*Lrc,len(Elements[1]))

        def PlaneWaveFocus(angles):

            T = np.meshgrid(f, drc * np.sin(np.deg2rad(angles[1])) / c)

            XX = np.sum(X * np.exp(-2j * np.pi *
                                   T[0] * T[1]), axis=1, keepdims=False)

            T = np.meshgrid(f, dtr * np.sin(np.deg2rad(angles[0])) / c)

            XX = np.sum(XX * np.exp(-2j * np.pi *
                                    T[0] * T[1]), axis=0, keepdims=False)

            x = 2*ifft(XX,2*len(XX))

            return x[0:L]

        if isinstance(Angles, tuple):

            return np.array([[PlaneWaveFocus((ta, ra))
                              for ra in Angles[1]] for ta in Angles[0]])

        else:

            return np.array([PlaneWaveFocus((ta, ta)) for ta in Angles])

    def GetContactDelaysOnLine(self, x, y, c):

        if c is None:

            c = self.Velocity

        xn = np.linspace(-(self.NumberOfElements-1)*self.Pitch*0.5, (self.NumberOfElements-1)*self.Pitch*0.5, self.NumberOfElements)

        delays = [np.sqrt((x - xn[n])**2 + y**2)/c for n in range(self.NumberOfElements)]

        self.Delays = (delays,delays)

    def GetContactDelays(self, xrng, yrng, c):

        if c is None:

            c = self.Velocity

        xn = np.linspace(-(self.NumberOfElements-1)*self.Pitch*0.5, (self.NumberOfElements-1)*self.Pitch*0.5, self.NumberOfElements)

        x,y = np.meshgrid(xrng, yrng)

        delays = [np.sqrt((x - xn[n])**2 + y**2)/c for n in range(self.NumberOfElements)]

        self.Delays = (delays,delays)


        self.xRange = xrng.copy()
        self.yRange = yrng.copy()

    def GetContactBackwallDelays(self, xrng, yrng, c, Th):

        if c is None:

            c = self.Velocity

        xn = np.linspace(-(self.NumberOfElements-1)*self.Pitch*0.5, (self.NumberOfElements-1)*self.Pitch*0.5, self.NumberOfElements)

        x,y = np.meshgrid(xrng, yrng)

        delays = [np.sqrt((x - xn[n])**2 + (2*Th-y)**2)/c for n in range(self.NumberOfElements)]

        self.Delays = (delays,delays)


        self.xRange = xrng.copy()
        self.yRange = yrng.copy()

    def GetContactFocusOnReceptionDelays(self, c, angle=0):

        angle = angle*np.pi/180
        N = self.NumberOfElements
        p = self.Pitch

        x = self.xRange
        y=self.yRange

        X,Y = np.meshgrid(self.xRange,self.yRange)

        delays = []

        for n in range(N):

            d = Y/(c*np.cos(angle))

            for i in range(len(self.yRange)):

                for j in range(len(self.xRange)):

                    if ((x[j]<y[i]*np.tan(angle)) or (x[j]>(N*p + y[i]*np.tan(angle)))):

                        d[i][j] = np.nan

            delays.append(d)

        self.Delays = (delays,delays)

    def GetZeroDegreeWedgeFocusOnReceptionDelays(self,c,angle=0):

        h = self.WedgeParameters['Height']
        cw = self.WedgeParameters['Velocity']
        N = self.NumberOfElements

        X,Y = np.meshgrid(self.xRange,self.yRange)
        delays = []

        for n in range(N):

            d = Y/c + h/cw

            delays.append(d)

        self.Delays = (delays,delays)

    def GetWedgeFocusOnReceptionDelays(self,c,angle):

        phiw = self.WedgeParameters['Angle']*np.pi/180
        p = self.Pitch
        h = self.WedgeParameters['Height']
        cw = self.WedgeParameters['Velocity']

        x = self.xRange
        y = self.yRange

        X,Y = np.meshgrid(self.xRange,self.yRange)

        delays = []

        phiR = angle*np.pi/180
        Phii = np.arcsin((cw/c)*np.sin(phiR))

        for n in range(N):

            d = Y/(c*np.cos(phiR)) + ((X - Y*np.tan(phiR))*np.tan(phiw) + h)/(cw*np.cos(phii)*np.tan(phiw)*(np.tan(phii)+np.tan(phiw)))

            for i in range(len(self.yRange)):

                for j in range(len(self.xRange)):

                    if ((x[j]<h*np.tan(phii)+y[i]*tan(phiR)) or (x[j]>((h + N*p*np.sin(phiw))*np.tan(phii)+ N*p*np.cos(phiw) + y[i]*np.tan(phiR)))):

                        d[i][j] = np.nan

            delays.append(d)

            self.Delays = (delays,delays)

    def GetContactCorrections(self, x,y,amplitude,sensitivity=None, isongrid=False):

        from scipy.interpolate import griddata

        if isongrid:

            xyi = np.meshgrid(x, y)

            xyi = (xyi[0].flatten(), xyi[1].flatten())

        else:

            xyi = (x.flatten(), y.flatten())

        self.AmplitudeCorrection = []

        for n in range(self.NumberOfElements):

            xyp = np.meshgrid(self.xRange - n * self.Pitch, self.yRange)

            if sensitivity is not None:

                A = griddata(xyi,sensitivity[n]*amplitude.flatten(),(xyp[0].flatten(),xyp[1].flatten()),fill_value=np.nan,method='linear').reshape(xyp[0].shape)

            else:

                A = griddata(xyi,amplitude.flatten(),(xyp[0].flatten(),xyp[1].flatten()),fill_value=np.nan,method='linear').reshape(xyp[0].shape)

            ind = np.where(np.isfinite(A[0,:]))[0]

            A[:,0:ind[0]]=A[:,ind[0]].reshape((-1,1))

            A[:,ind[-1]::]=A[:,ind[-1]].reshape((-1,1))

            self.AmplitudeCorrection.append(A)

    def KeepElements(self, Elements):

        for i in range(len(self.AScans)):

            self.AScans[i] = np.take(np.take(self.AScans[i],Elements,axis=0),Elements,axis=1)

        if self.ProbeDelays is not None:

            self.ProbeDelays = np.take(np.take(self.ProbeDelays,Elements,axis=0),Elements,axis=1)

        self.NumberOfElements = len(Elements)

    def FitInterfaceLine(self, ScanIndex, angrng, gate, c):
        """
        gate specified in terms of mm in medium with Velocity c
        """
        angles = np.arange(angrng[0], angrng[1], angrng[2])

        X = np.abs(
            self.PlaneWaveSweep(
                ScanIndex, angles, c))[
            :, int(
                np.round(
                    2 * gate[0] * self.SamplingFrequency / c)):int(
                        np.round(
                            2 * gate[1] * self.SamplingFrequency / c))]

        imax = np.unravel_index(X.argmax(), X.shape)

        angmax = -angles[imax[0]] * np.pi / 180.

        hmax = gate[0] + imax[1] * c / (2 * self.SamplingFrequency)

        h0 = hmax / (np.cos(angmax))

        def h(x): return np.tan(angmax) * x + h0 - self.Pitch * \
            self.NumberOfElements * np.tan(angmax) / 2

        return h

    def GetAdaptiveDelays(self, ScanIndex, xrng, yrng, c, captracetype='TFM',Lw=10):

        from scipy.optimize import minimize_scalar, minimize
        from scipy.interpolate import interp1d,griddata
        from scipy.signal import convolve, decimate
        from matplotlib.pylab import plot,show

        xn = np.linspace(-0.5*(self.NumberOfElements-1)*self.Pitch,0.5*(self.NumberOfElements-1)*self.Pitch,self.NumberOfElements)

        if captracetype=='TFM':

            self.GetContactDelays(xrng[0], yrng[0], c[0])

            I = self.ApplyTFM(ScanIndex)

            dh = yrng[0][1] - yrng[0][0]

            hgrid = np.argmax(np.abs(I),axis=0)*dh + yrng[0][0]

            hgrid = decimate(hgrid,Lw)

            h = interp1d(xrng[0][0::Lw], hgrid, bounds_error=False)


        elif captracetype=='diag':

            t0 = int(self.SamplingFrequency*2*yrng[0][0]/c[0])


            hgrid = np.argmax(np.array([np.abs(self.AScans[ScanIndex][n,n,:]) for n in range(self.NumberOfElements)]).transpose(),axis=0)*0.5*c[0]/self.SamplingFrequency

            h = interp1d(xrng[0], hgrid, bounds_error=False)



        def f(x, X, Y, n):

            return np.sqrt((x-xn[n])**2 + (h(x))**2)/c[0] + np.sqrt((X - x)**2 + (Y - h(x))**2)/c[1]

        def DelayMin(x,y,n):

            if (y < h(x)):

                T = np.nan

            elif (xn[n]!=x):

                bnds = (min([xn[n],x]),max([xn[n],x]))


                T = minimize_scalar(f,bnds,args=(x,y,n),tol=1e-2).fun


            elif xn[n]==x:

                T = f(x,x,y,n)

            return T

        DelayMin = np.vectorize(DelayMin ,excluded=['n'])

        x,y = np.meshgrid(xrng[1],yrng[1])

        d = [DelayMin(x,y,n) for n in range(self.NumberOfElements)]

        self.Delays = (d,d)

        self.xRange = xrng[1]

        self.yRange = yrng[1]

    def FilterByAngle(self, ScanIndex, filtertype, angle, FWHM, c):

        L = self.AScans[ScanIndex].shape[2]

        X = fftshift(
            fftn(
                np.real(
                    self.AScans[ScanIndex]), s=(
                    self.NumberOfElements, L), axes=(
                    0, 2)), axes=(0))


        X = X[:, :, 0:int(np.floor(X.shape[2] / 2) + 1)]

        kx = 2 * np.pi * np.linspace(-1 / (2 * self.Pitch),
                                     1 / (2 * self.Pitch), X.shape[0]).reshape(-1, 1)

        w = 2 * np.pi * np.linspace(0., self.SamplingFrequency /
                                    2, X.shape[2]).reshape(1, -1)

        th = np.arcsin(c * kx / w)

        th = np.repeat(
            th.reshape(
                (kx.shape[0],
                 1,
                 w.shape[1])),
            X.shape[1],
            axis=1) * 180 / np.pi

        alpha = ((2.35482)**2) / (2 * FWHM**2)

        FilterFunction = {
            'Band': np.exp(-alpha * (th - angle)**2), 'Notch': 1. - np.exp(-alpha * (th - angle)**2)}

        H = np.nan_to_num(FilterFunction[filtertype]).astype(type(X[0, 0, 0]))

        X = H * X

        X = fftshift(X, axes=(0))

        return 2 * ifftn(X, s=(X.shape[0], L), axes=(0, 2))

    def ApplyTFM(self, ScanIndex, Elements=None, FilterParams=None, Normalize=False, OnLine=False):

        if FilterParams is None:

            a = self.AScans[ScanIndex]

        else:

            a = self.FilterByAngle(
                ScanIndex,
                FilterParams[0],
                FilterParams[1],
                FilterParams[2],
                FilterParams[3])


        L = a.shape[2]

        t = np.linspace(0.,L-1,L)/self.SamplingFrequency

        if Elements is None:

            Elements = (range(self.NumberOfElements), range(self.NumberOfElements))

        def ElementFocus(m,n):

            I = np.interp((self.Delays[0][m]+self.Delays[1][n]).flatten(),t,a[Elements[0][m],Elements[1][n],:])

            np.nan_to_num(I,copy=False)

            return I

        if OnLine:

            I = reduce(lambda x,y: x+y, (ElementFocus(m,n) for m in range(len(Elements[0])) for n in range(len(Elements[1]))))

        else:

            I = reduce(lambda x,y: x+y, (ElementFocus(m,n) for m in range(len(Elements[0])) for n in range(len(Elements[1])))).reshape((len(self.yRange),len(self.xRange)))


        if Normalize:

            I/np.amax(np.abs(I))

        return I

    def FocusOnReception(self, ScanIndex, Elements=None, Normalize=False):

        a = self.AScans[ScanIndex]

        L = a.shape[2]

        t = np.linspace(0.,L-1,L)/self.SamplingFrequency

        if Elements is None:

            Elements = range(self.NumberOfElements)

        def ElementFocus(n):

            I = np.interp((self.Delays[0]+self.Delays[1][n]).flatten(),t,a[0,Elements[n],:])

            np.nan_to_num(I,copy=False)

            return I


        I = reduce(lambda x,y: x+y, (ElementFocus(n) for n in range(len(Elements)))).reshape((len(self.yRange),len(self.xRange)))
        # I = reduce(lambda x,y: x+y, (ElementFocus(n) for n in range(len(Elements[1]))))

        if Normalize:

            I/np.amax(np.abs(I))

        return I
