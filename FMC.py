from functools import reduce
import numpy as np
from numpy.fft import rfft, ifft, fftn, ifftn, fftshift
from pathos.multiprocessing import ProcessingPool
import os
import multiprocessing

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

    print(W)

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


class LinearCapture:

    def __init__(self, fs, scans, p, N, probedelays=None, WedgeParameters=None):

        import copy

        self.SamplingFrequency = fs
        self.Pitch = p
        # self.Velocity = c
        self.NumberOfElements = N

        if probedelays is None:

            self.ProbeDelays = np.zeros((N, N))

        else:

            self.ProbeDelays = probedelays


        self.AScans = copy.deepcopy(scans)

        self.AScans = [a.astype(np.float)/32768. for a in self.AScans]

        self.AmplitudeCorrection = None

        self.WedgeParameters = WedgeParameters

        GetDelays = {}
        GetDelays['DirectDirect'] = lambda c, offset: (GetWedgeDelays(c[0],key='send',offset[0]),GetWedgeDelays(c[1],key='rec',offset[1]))
        GetDelays['BackwallDirect'] = lambda c,Th,offset: (GetWedgeBackWallDelays((c[0],c[1]),Th,key='send',offset[0]), GetWedgeDelays(c[2],key='rec',offset[1]))
        GetDelays['BackWallBackWall'] = lambda c,Th,offset: (GetWedgeBackWallDelays((c[0],c[1]),Th,key='send',offset[0]),GetWedgeBackWallDelays((c[2],c[3]),Th,key='rec',offset[1]))
        self.GetDelays = GetDelays


    def SetRectangularGrid(xstart,xend,ystart,yend,xres,yres):

        Nx = np.floor((xend - xstart)/xres) + 1
        Ny = np.floor((yend - ystart)/yres) + 1

        x = np.linspace(xstart,send,Nx)
        y = np.linspace(ystart,send,Ny)

        self.xRange = x
        self.yRange = y

        x,y = np.meshgrid(x,y)

        self.SendDelays = [np.zeros(x.shape) for n in range(self.NumberOfElements)]
        self.RecDelays = [np.zeros(x.shape) for n in range(self.NumberOfElements)]

    def GetWedgeDelays(self, c, key, Offset = 0):

        from scipy.optimize import minimize_scalar,minimize
        # from scipy.optimize import brentq

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


        x,y = np.meshgrid(self.xRange + Offset,self.yRange)

        ComputeDelays = np.vectorize(f,excluded=['n'])

        if key is 'send':

            self.SendDelays = [ComputeDelays(x,y,n) for n in range(self.NumberOfElements)]

        else:

            self.RecDelays = [ComputeDelays(x,y,n) for n in range(self.NumberOfElements)]

    def GetWedgeBackWallDelays(self, c, Th, key, Offset):

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

                t = sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x0)**2)/cw + sqrt((Th + Y)**2 + (X - x1)**2)/c2 + sqrt(Th**2 + (-x0 + x1)**2)/c1

                dtdx = [-(cphi*n*p - x0)/(cw*sqrt((h + n*p*sphi)**2 + (cphi*n*p - x0)**2)) + (x0 - x1)/(c1*sqrt(Th**2 + (x0 - x1)**2)),-(X - x1)/(c2*sqrt((Th + Y)**2 + (X - x1)**2)) - (x0 - x1)/(c1*sqrt(Th**2 + (x0 - x1)**2))]

                return t,np.array(dtdx)

            def delays(X,Y,n):

                bnds = ((n*p*cphi,X),(n*p*cphi,X))

                xi = (0.5*(bnds[0][1] + bnds[0][0]),0.5*(bnds[0][1] + bnds[0][0]))

                res = minimize(f,xi,args=(X,Y,n),method='BFGS',jac='True')

                return res.fun

            if key is 'send':

                self.SendDelays =  np.array([[[delays(X,Y,n) for Y in self.yRange] for X in self.xRange + Offset] for n in range(self.NumberOfElements)])

            else:

                self.RecDelays =  np.array([[[delays(X,Y,n) for Y in self.yRange] for X in self.xRange + Offset] for n in range(self.NumberOfElements)])


    def ProcessScans(self, zeropoints=20, bp=10, normalize=True):

        from scipy.signal import detrend, hilbert
        from numpy.linalg import norm

        L = self.AScans[0].shape[2]

        d = np.round(self.ProbeDelays*self.SamplingFrequency).astype(int)

        dmax = np.amax(d)

        if dmax<zeropoints:

            for i in range(len(self.AScans)):

                for m in range(self.NumberOfElements):

                    for n in range(self.NumberOfElements):

                        self.AScans[i][m,n,0:zeropoints-d[m,n]] = 0.

                self.AScans[i] = hilbert(detrend(self.AScans[i], bp=list(np.arange(0, L, bp).astype(int))))

                if normalize:

                    self.AScans[i] = self.AScans[i]/norm(self.AScans[i])


    def SetGridforPipe(Radious,Thickness,offset,xres,yres,convex = True):

        ymax = -offset/2

        if convex:

            ymin = -(offset + Thickness)

        else:

            ymin = -offset - Radious + np.sqrt((Radious - Thickness)**2 - (0.5*(self.NumberofElements - 1))**2)

        Nx = np.floor((self.NumberofElements * self.Pitch)/xres) + 1
        Ny = np.floor((ymax - ymin)/yres) + 1

        x = np.linspace(-0.5*(self.NumberofElements - 1),0.5*(self.NumberofElements - 1),Nx)
        y = np.linspace(ymin,ymax,Ny)

        x,y = np.meshgrid(x,y)

        self.xRange = x
        self.yRange = y

        self.Delays = [np.zeros(x.shape) for n in range(self.NumberOfElements)]


    def GetCurvedSurfaceDelays(self, Radious, Thickness, Offset, c, Convexin = False, Convexout = False):

        # For the case of center of the array aligns with the center of the curvature.

        R = Radious
        h = Thickness
        d = Offset

        from scipy.optimize import minimize_scalar,minimize

        p = self.Pitch
        cw = self.WedgeParameters['Velocity']
        N = self.NumberOfElements

        phi = np.arcsin(0.5*p*(1-N)/(R-h))
        phi = np.array(np.linspace(-phi,phi,10))
        r = np.array(np.linspace(R-h,R,10))
        xm = r*np.sin(phi)

        def f(x,X,Y,n):

            t = np.sqrt((-0.5*p*(-N + 2*n + 1) + x)**2 + (-R - d + np.sqrt(R**2 - x**2))**2)/cw + ((-x + X)**2 + (R + d + Y - np.sqrt(R**2 - x**2))**2)/c

            dtdx = (c*(x*(R + d - np.sqrt(R**2 - x**2)) + np.sqrt(R**2 - x**2)*(-0.5*p*(-N + 2*n + 1) + x)) + 2*cw*(x*(R + d + Y - np.sqrt(R**2 - x**2)) + np.sqrt(R**2 - x**2)*(x - X))*np.sqrt((0.5*p*(-N + 2*n + 1) - x)**2 + (R + d - np.sqrt(R**2 - x**2))**2))/(c*cw*np.sqrt(R**2 - x**2)*np.sqrt((0.5*p*(-N + 2*n + 1) - x)**2 + (R + d - np.sqrt(R**2 - x**2))**2))

            return t, dtdx

        def g(x,X,Y,n):

            t = np.sqrt((-d - np.sqrt(-x**2 + (R - h)**2))**2 + (-0.5*p*(-N + 2*n + 1) + x)**2)/cw + np.sqrt((X - x)**2 + (Y + d + np.sqrt(-x**2 + (R - h)**2))**2)/c

            dtdx = (c*(-x*(d + np.sqrt(-x**2 + (R - h)**2)) + np.sqrt(-x**2 + (R - h)**2)*(-0.5*p*(-N + 2*n + 1) + x))*np.sqrt((X - x)**2 + (Y + d + np.sqrt(-x**2 + (R - h)**2))**2) + cw*(-x*(Y + d + np.sqrt(-x**2 + (R - h)**2)) + (-X + x)*np.sqrt(-x**2 + (R - h)**2))*np.sqrt((d + np.sqrt(-x**2 + (R - h)**2))**2 + (0.5*p*(-N + 2*n + 1) - x)**2))/(c*cw*np.sqrt(-x**2 + (R - h)**2)*np.sqrt((X - x)**2 + (Y + d + np.sqrt(-x**2 + (R - h)**2))**2)*np.sqrt((d + np.sqrt(-x**2 + (R - h)**2))**2 + (0.5*p*(-N + 2*n + 1) - x)**2))

            return t, dtdx

        def l(x,X,Y,n):

            t = np.sqrt((d - np.sqrt(-x**2 + (R - h)**2))**2 + (-0.5*p*(-N + 2*n + 1) + x)**2)/cw + np.sqrt((X - x)**2 + (Y - d + np.sqrt(-x**2 + (R - h)**2))**2)/c

            dtdx = (c*(x*(d - np.sqrt(-x**2 + (R - h)**2)) + np.sqrt(-x**2 + (R - h)**2)*(-0.5*p*(-N + 2*n + 1) + x))*np.sqrt((X - x)**2 + (Y - d + np.sqrt(-x**2 + (R - h)**2))**2) + cw*(-x*(Y - d + np.sqrt(-x**2 + (R - h)**2)) + (-X + x)*np.sqrt(-x**2 + (R - h)**2))*np.sqrt((d - np.sqrt(-x**2 + (R - h)**2))**2 + (0.5*p*(-N + 2*n + 1) - x)**2))/(c*cw*np.sqrt(-x**2 + (R - h)**2)*np.sqrt((X - x)**2 + (Y - d + np.sqrt(-x**2 + (R - h)**2))**2)*np.sqrt((d - np.sqrt(-x**2 + (R - h)**2))**2 + (0.5*p*(-N + 2*n + 1) - x)**2))

            return t, dtdx

        def delays(X,Y,n):

            bnds = (0.5*p*(1-N+2*n), X)

            xi = 0.5*(bnds[1] + bnds[0])

            if Convexin:

                res = minimize(l,xi,args=(X,Y,n),method='BFGS',jac='True')

                return res.fun

            elif Convexout:

               res = minimize(g,xi,args=(X,Y,n),method='BFGS',jac='True')

               return res.fun

            else:

               res = minimize(f,xi,args=(X,Y,n),method='BFGS',jac='True')

               x = res.x
               y = -(d+R)+np.sqrt(R**2-x**2)
               m = (Y-y)/(X-x)
               a = -m*X+Y+d+R

               C = np.zeros(3)
               C[0] = 1+m**2
               C[1] = 2*m*a
               C[2] = a**2 - (R-h)**2

               A = np.roots(C)

               if (np.isreal(A[0])) or (np.isreal(A[1])):

                   return nan

               else:

                   return res.fun


        self.Delays = np.array([[[delays(X,Y,n) for Y in self.yRange] for X in xm] for n in range(N)])

        self.Delays =  [self.Delays[n] + np.array([[delays(X,Y,n) if condition else 0 for Y in yrng] if condition else 0 for X in xrng]) for n in range(self.NumberOfElements)]

        self.xRange = xm.copy()
        self.yRange = ym.copy()

    def GetCurvedSurfaceBackwallDelays(self, Radious, Thickness, Offset, c, Convexin = False, Convexout = False):

        # For the case of center of the array aligns with the center of the curvature.

        R = Radious
        h = Thickness
        d = Offset
        c1 = c[0]
        c2 = c[1]

        from scipy.optimize import minimize_scalar,minimize

        p = self.Pitch
        cw = self.WedgeParameters['Velocity']
        N = self.NumberOfElements

        phi = np.arcsin(0.5*p*(1-N)/(R-h))
        phi = array(np.linspace(-phi,phi,10))
        r = array(np.linspace(R-h,R,10))
        xm = r*np.sin(phi)

        def f(x,X,Y,n):

            x0,x1 = x[0],x[1]

            t = sqrt((-0.5*p*(-N + 2*n + 1) + x0)**2 + (-R - d + sqrt(R**2 - x0**2))**2)/cw + sqrt((X - x1)**2 + (R + Y + d - sqrt(-x1**2 + (R - h)**2))**2)/c2 + sqrt((-x0 + x1)**2 + (-sqrt(R**2 - x0**2) + sqrt(-x1**2 + (R - h)**2))**2)/c1

            dtdx = [(c1*(x0*(R + d - sqrt(R**2 - x0**2)) + sqrt(R**2 - x0**2)*(-0.5*p*(-N + 2*n + 1) + x0))*sqrt((x0 - x1)**2 + (sqrt(R**2 - x0**2) - sqrt(-x1**2 + (R - h)**2))**2) + cw*(-x0*(sqrt(R**2 - x0**2) - sqrt(-x1**2 + (R - h)**2)) + sqrt(R**2 - x0**2)*(x0 - x1))*sqrt((0.5*p*(-N + 2*n + 1) - x0)**2 + (R + d - sqrt(R**2 - x0**2))**2))/(c1*cw*sqrt(R**2 - x0**2)*sqrt((x0 - x1)**2 + (sqrt(R**2 - x0**2) - sqrt(-x1**2 + (R - h)**2))**2)*sqrt((0.5*p*(-N + 2*n + 1) - x0)**2 + (R + d - sqrt(R**2 - x0**2))**2)),(c1*(x1*(R + Y + d - sqrt(-x1**2 + (R - h)**2)) + (-X + x1)*sqrt(-x1**2 + (R - h)**2))*sqrt((x0 - x1)**2 + (sqrt(R**2 - x0**2) - sqrt(-x1**2 + (R - h)**2))**2) + c2*(x1*(sqrt(R**2 - x0**2) - sqrt(-x1**2 + (R - h)**2)) + (-x0 + x1)*sqrt(-x1**2 + (R - h)**2))*sqrt((X - x1)**2 + (R + Y + d - sqrt(-x1**2 + (R - h)**2))**2))/(c1*c2*sqrt(-x1**2 + (R - h)**2)*sqrt((X - x1)**2 + (R + Y + d - sqrt(-x1**2 + (R - h)**2))**2)*sqrt((x0 - x1)**2 + (sqrt(R**2 - x0**2) - sqrt(-x1**2 + (R - h)**2))**2))]

            return t,array(dtdx)

        def g(x,X,Y,n):

            x0,x1 = x[0],x[1]

            t = sqrt((-d - sqrt(-x0**2 + (R - h)**2))**2 + (-0.5*p*(-N + 2*n + 1) + x0)**2)/cw + sqrt((X - x1)**2 + (Y + d + sqrt(R**2 - x1**2))**2)/c2 + sqrt((-x0 + x1)**2 + (-sqrt(R**2 - x1**2) + sqrt(-x0**2 + (R - h)**2))**2)/c1

            dtdx = [(c1*(-x0*(d + sqrt(-x0**2 + (R - h)**2)) + sqrt(-x0**2 + (R - h)**2)*(-0.5*p*(-N + 2*n + 1) + x0))*sqrt((x0 - x1)**2 + (sqrt(R**2 - x1**2) - sqrt(-x0**2 + (R - h)**2))**2) + cw*(x0*(sqrt(R**2 - x1**2) - sqrt(-x0**2 + (R - h)**2)) + (x0 - x1)*sqrt(-x0**2 + (R - h)**2))*sqrt((d + sqrt(-x0**2 + (R - h)**2))**2 + (0.5*p*(-N + 2*n + 1) - x0)**2))/(c1*cw*sqrt(-x0**2 + (R - h)**2)*sqrt((d + sqrt(-x0**2 + (R - h)**2))**2 + (0.5*p*(-N + 2*n + 1) - x0)**2)*sqrt((x0 - x1)**2 + (sqrt(R**2 - x1**2) - sqrt(-x0**2 + (R - h)**2))**2)),(c1*(-x1*(Y + d + sqrt(R**2 - x1**2)) + sqrt(R**2 - x1**2)*(-X + x1))*sqrt((x0 - x1)**2 + (sqrt(R**2 - x1**2) - sqrt(-x0**2 + (R - h)**2))**2) + c2*(-x1*(sqrt(R**2 - x1**2) - sqrt(-x0**2 + (R - h)**2)) + sqrt(R**2 - x1**2)*(-x0 + x1))*sqrt((X - x1)**2 + (Y + d + sqrt(R**2 - x1**2))**2))/(c1*c2*sqrt(R**2 - x1**2)*sqrt((X - x1)**2 + (Y + d + sqrt(R**2 - x1**2))**2)*sqrt((x0 - x1)**2 + (sqrt(R**2 - x1**2) - sqrt(-x0**2 + (R - h)**2))**2))]

            return t, array(dtdx)

        def l(x,X,Y,n):

            x0,x1 = x[0],x[1]

            t = sqrt((d - sqrt(-x0**2 + (R - h)**2))**2 + (-0.5*p*(-N + 2*n + 1) + x0)**2)/cw + sqrt((X - x1)**2 + (Y - d + sqrt(R**2 - x1**2))**2)/c2 + sqrt((-x0 + x1)**2 + (-sqrt(R**2 - x1**2) + sqrt(-x0**2 + (R - h)**2))**2)/c1

            dtdx = [(c1*(x0*(d - sqrt(-x0**2 + (R - h)**2)) + sqrt(-x0**2 + (R - h)**2)*(-0.5*p*(-N + 2*n + 1) + x0))*sqrt((x0 - x1)**2 + (sqrt(R**2 - x1**2) - sqrt(-x0**2 + (R - h)**2))**2) + cw*(x0*(sqrt(R**2 - x1**2) - sqrt(-x0**2 + (R - h)**2)) + (x0 - x1)*sqrt(-x0**2 + (R - h)**2))*sqrt((d - sqrt(-x0**2 + (R - h)**2))**2 + (0.5*p*(-N + 2*n + 1) - x0)**2))/(c1*cw*sqrt(-x0**2 + (R - h)**2)*sqrt((d - sqrt(-x0**2 + (R - h)**2))**2 + (0.5*p*(-N + 2*n + 1) - x0)**2)*sqrt((x0 - x1)**2 + (sqrt(R**2 - x1**2) - sqrt(-x0**2 + (R - h)**2))**2)),(c1*(-x1*(Y - d + sqrt(R**2 - x1**2)) + sqrt(R**2 - x1**2)*(-X + x1))*sqrt((x0 - x1)**2 + (sqrt(R**2 - x1**2) - sqrt(-x0**2 + (R - h)**2))**2) + c2*(-x1*(sqrt(R**2 - x1**2) - sqrt(-x0**2 + (R - h)**2)) + sqrt(R**2 - x1**2)*(-x0 + x1))*sqrt((X - x1)**2 + (Y - d + sqrt(R**2 - x1**2))**2))/(c1*c2*sqrt(R**2 - x1**2)*sqrt((X - x1)**2 + (Y - d + sqrt(R**2 - x1**2))**2)*sqrt((x0 - x1)**2 + (sqrt(R**2 - x1**2) - sqrt(-x0**2 + (R - h)**2))**2))]

    def ReverseElements(self):

        self.AScans = [a[::-1,::-1,:] for a in self.AScans]


        # Lpad = NextPow2(np.round((L + np.amax(self.ProbeDelays)*self.SamplingFrequency - 1)))
        #
        # Lpad = int(
        #     np.round(
        #         (L +
        #          np.amax(
        #              self.ProbeDelays) *
        #             self.SamplingFrequency -
        #             1)))
        #
        # f = np.linspace(0., self.SamplingFrequency / 2, np.floor(Lpad / 2) + 1)
        #
        # f = f.reshape((1, 1, len(f)))
        #
        # D = np.exp(
        #     2j * np.pi * np.repeat(self.ProbeDelays[:, :, np.newaxis], f.shape[2], 2) * f)
        #
        # A = np.repeat(self.ElementAmplitudes[:, :, np.newaxis], f.shape[2], 2).astype(
        #     np.complex64)
        #
        # for i in range(len(self.AScans)):
        #
        #     if zeropoints != 0:
        #         self.AScans[i][:, :, 0:zeropoints] = 0.0
        #
        #     self.AScans[i] = detrend(
        #         self.AScans[i], bp=list(
        #             np.linspace(
        #                 0, L - 1, bp).astype(int)))
        #
        #     X = rfft(self.AScans[i], n=Lpad)
        #
        #     self.AScans[i] = self.AScans[i].astype(np.complex64)
        #
        #     self.AScans[i] = 2 * ifft((X / A) * D, n=Lpad)[:, :, 0:L]
        #
        # if T0 is not None:
        #
        #     Npad = int(round(T0*self.SamplingFrequency))
        #
        #     zpad = np.zeros(Npad,dtype=np.complex64)
        #
        #     self.AScans = [np.concatenate((zpad, a)) for a in self.AScans]


    def PlaneWaveSweep(self, ScanIndex, Angles, Elements, c):


        X = np.real(self.AScans[ScanIndex][Elements[0],Elements[1],:])

        L = X.shape[2]

        X = rfft(X, 2*L)

        f = np.linspace(0, self.SamplingFrequency / 2, X.shape[2])

        Ltr = self.Pitch*(len(Elements[0])-1)

        Lrc = self.Pitch*(len(Elements[1])-1)

        dtr = np.linspace(-0.5*Ltr,0.5*Ltr,len(Elements[0]))

        drc = np.linspace(-0.5*Lrc,0.5*Lrc,len(Elements[1]))


        def PlaneWaveFocus(angles):

            T = np.meshgrid(f, drc * np.sin(angles[1]) / c)

            XX = np.sum(X * np.exp(-2j * np.pi *
                                   T[0] * T[1]), axis=1, keepdims=False)

            T = np.meshgrid(f, dtr * np.sin(angles[0]) / c)

            XX = np.sum(XX * np.exp(-2j * np.pi *
                                    T[0] * T[1]), axis=0, keepdims=False)

            x = ifft(XX)

            return x[0:L]

        if isinstance(Angles, tuple):

            return np.array([[PlaneWaveFocus((ta, ra))
                              for ra in Angles[1]] for ta in Angles[0]])

        else:

            return np.array([PlaneWaveFocus((ta, ta)) for ta in Angles])

    def GetContactDelays(self, xrng, yrng, c):

        if c is None:

            c = self.Velocity

        # self.Delays = [[[np.sqrt((x - n * self.Pitch)**2 + y**2) / c for y in yrng]
        #                 for x in xrng] for n in range(self.NumberOfElements)]

        xn = np.linspace(-(self.NumberOfElements-1)*self.Pitch*0.5, (self.NumberOfElements-1)*self.Pitch*0.5, self.NumberOfElements)

        x,y = np.meshgrid(xrng, yrng)


        # self.Delays = [np.sqrt((x - n*self.Pitch)**2 + y**2)/c for n in range(self.NumberOfElements)]

        self.Delays = [np.sqrt((x - xn[n])**2 + y**2)/c for n in range(self.NumberOfElements)]


        # self.GetDelayIndices()


        self.xRange = xrng.copy()

        self.yRange = yrng.copy()


    def GetContactCorrections(self, x,y,amplitude,sensitivity=None, isongrid=False):

        from scipy.interpolate import griddata

        if isongrid:

            xyi = np.meshgrid(x, y)

            xyi = (xyi[0].flatten(), xyi[1].flatten())

        else:

            xyi = (x.flatten(), y.flatten())


        # if sensitivity is not None:
        #
        #     sensitivity = np.sqrt(sensitivity/np.amax(sensitivity))

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

            self.AScans[i] = np.take(
                np.take(
                    self.AScans[i],
                    Elements,
                    axis=0),
                Elements,
                axis=1)

        if self.ProbeDelays is not None:

            self.ProbeDelays = np.take(
                np.take(
                    self.ProbeDelays,
                    Elements,
                    axis=0),
                Elements,
                axis=1)

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

    def GetAdaptiveDelays(self, ScanIndex, xrng, yrng, cw, cs):

        from scipy.optimize import minimize_scalar, minimize
        from scipy.interpolate import interp1d
        from scipy.signal import convolve

        self.GetContactDelays(xrng, yrng[0], cw)

        I = self.ApplyTFM(ScanIndex)

        dh = yrng[0][1] - yrng[0][0]

        hgrid = np.argmax(np.abs(I), axis=0) * dh + yrng[0][0]

        Lw = int(np.round(self.Pitch/(xrng[1]-xrng[0])))

        w = np.ones(Lw)/Lw

        hgrid = convolve(hgrid,w,mode='same')

        h = interp1d(xrng, hgrid, bounds_error=False)

        def f(x, X, Y, n):
            return np.sqrt((x - n * self.Pitch)**2 + h(x) ** 2)/cw + np.sqrt((X - x)**2 + (Y - h(x))**2)/cs


        def DelayMin(n):

            return np.array([[np.float(minimize(f,0.5*(x+self.Pitch*n),args=(x,y,n),method='BFGS',tol=1e-2,options={'gtol':1e-3,'maxiter':20,'eps':1e-5}).fun) if y >= h(x) else np.nan for y in yrng[1]] for x in xrng]).transpose()


        self.Delays = [DelayMin(n) for n in range(self.NumberOfElements)]

        self.xRange = xrng

        self.yRange = yrng

        return hgrid,h

    def FilterByAngle(self, ScanIndex, filtertype, angle, FWHM, c):

        L = self.AScans[ScanIndex].shape[2]

        # Lpad = NextPow2(L)

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

    def ApplyTFM(self, ScanIndex, Elements=None, FilterParams=None, stablecoeff=1e-4, Normalize=False):

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

        if self.AmplitudeCorrection is None:

            def ElementFocus(m,n):

                I = np.interp((self.Delays[m]+self.Delays[n]).flatten(),t,a[m,n,:])

                np.nan_to_num(I,copy=False)

                return I

        else:

            def ElementFocus(m,n):

                I = np.interp((self.Delays[m]+self.Delays[n]).flatten(),t,a[m,n,:])

                I = I/((self.AmplitudeCorrection[m]*self.AmplitudeCorrection[n]+stablecoeff).flatten())

                np.nan_to_num(I,copy=False)

                return I


        I = reduce(lambda x,y: x+y, (ElementFocus(m,n) for m in range(self.NumberOfElements) for n in range(self.NumberOfElements))).reshape(self.Delays[0].shape)

        if Normalize:

            I/np.amax(np.abs(I))

        return I
