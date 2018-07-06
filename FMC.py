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


    def SetRectangularGrid(xstart,xend,ystart,end,xres,yres):

        Nx = np.floor((xend - xstart)/xres) + 1
        Ny = np.floor((yend - ystart)/yres) + 1

        x = np.linspace(xstart,send,Nx)
        y = np.linspace(ystart,send,Ny)

        x,y = np.meshgrid(x,y)

        self.xRange = x
        self.yRange = y

        self.Delays = [np.zeros(x.shape) for n in range(self.NumberOfElements)]

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




    def GetWedgeBackWallDelays(self, xrng, yrng, c, Th):

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



        self.Delays =  [self.Delays[n] + np.array([[delays(X,Y,n) for Y in yrng] if condition else 0 for X in xrng]) for n in range(self.NumberOfElements)]



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

            return t, array(dtdx)


        def delays(X,Y,n):

            bnds = ((0.5*p*(1-N+2*n), X),(0.5*p*(1-N+2*n), X))

            xi = (0.5*(bnds[1] + bnds[0]),0.5*(bnds[1] + bnds[0]))

            if Convexin:

                res = minimize(l,xi,args=(X,Y,n),method='BFGS',jac='True',bounds=bnds)

                x = res.x
                y0 = d - np.sqrt((R-h)**2-x[0]**2)
                y1 = d - np.sqrt(R**2-x[1]**2)
                m = (y1-y0)/(x[1]-x[0])
                a = -m*x[1] + y1 - d

            elif Convexout:

                res = minimize(g,xi,args=(X,Y,n),method='BFGS',jac='True',bounds=bnds)

                x = res.x
                y0 = -d - np.sqrt((R-h)**2-x[0]**2)
                y1 = -d - np.sqrt(R**2-x[1]**2)
                m = (y1-y0)/(x[1]-x[0])
                a = -m*x[1] + y1 + d

            else:

                res = minimize(f,xi,args=(X,Y,n),method='BFGS',jac='True',bounds=bnds)

                x = res.x
                y0 = -(d+R) + np.sqrt(R**2-x[0]**2)
                y1 = -(d+R) + np.sqrt((R-h)**2-x[1]**2)
                m = (y1-y0)/(x[1]-x[0])
                a = -m*x[1] + y1 + d + R


            C = np.zeros(3)
            C[0] = 1 + m**2
            C[1] = 2*m*a
            C[2] = a**2 - (R-h)**2

            A = np.roots(C)

            if (np.isreal(A[0])) or (np.isreal(A[1])):

                return nan

            else:

                return res.fun


        if Convexin:

            ym = d - r*np.cos(phi)

        elif Convexout:

           ym = -d-r*np.cos(phi)

        else:

           ym = r*(np.cos(phi)-1)-d

        ym = r*(np.cos(phi)-1)-d

        self.Delays = np.array([[[delays(X,Y,n) for Y in ym] for X in xm] for n in range(N)])

        self.xRange = xm.copy()
        self.yRange = ym.copy()





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

    def PlaneWaveSweep(self, ScanIndex, Angles, c):

        # from numpy import sum as asum

        d = self.Pitch * (self.NumberOfElements - 1)

        # d = self.Pitch*self.NumberOfElements

        d = np.linspace(-d / 2, d / 2, self.NumberOfElements)

        L = self.AScans[ScanIndex].shape[2]

        if isinstance(Angles, tuple):

            Angles = (Angles[0] * np.pi / 180., Angles[1] * np.pi / 180.)

            T = np.abs(np.sin(np.repeat((Angles[0].reshape(-1, 1) * Angles[1].reshape(1, -1))[
                       :, :, np.newaxis], len(d), 2)) * d.reshape((1, 1, len(d))) / c).flatten()

        else:

            Angles = Angles * np.pi / 180.

            T = np.abs(np.sin(np.repeat((Angles.reshape(-1, 1) * Angles.reshape(1, -1))
                                        [:, :, np.newaxis], len(d), 2)) * d.reshape((1, 1, len(d))) / c).flatten()

        Npad = int(np.round(self.SamplingFrequency * np.max(T)))

        Lpad = L + Npad - 1
        # X = rfft(np.concatenate((np.zeros((self.NumberOfElements, self.NumberOfElements, Npadstart)),
        #                       np.real(self.AScans[ScanIndex]), np.zeros((self.NumberOfElements,
        # self.NumberOfElements, Npadend))), axis=2), axis=2)

        X = rfft(np.real(self.AScans[ScanIndex]), Lpad)

        f = np.linspace(0, self.SamplingFrequency / 2, X.shape[2])

        def PlaneWaveFocus(angles):

            T = np.meshgrid(f, d * np.sin(angles[1]) / c)

            XX = np.sum(X * np.exp(-2j * np.pi *
                                   T[0] * T[1]), axis=1, keepdims=False)

            T = np.meshgrid(f, d * np.sin(angles[0]) / c)

            XX = np.sum(XX * np.exp(-2j * np.pi *
                                    T[0] * T[1]), axis=0, keepdims=False)

            x = ifft(XX, n=Lpad)

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

    # def GetDelayIndices(self):
    #
    #         self.DelayIndices = [[[int(np.round(self.Delays[n][ix][iy]*self.SamplingFrequency)) for iy in range(len(self.Delays[0][0]))] for ix in range(len(self.Delays[0]))] for n in range(len(self.Delays))]
    #

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




    def GetWedgeDelays(self, xrng, yrng, c):

        from scipy.optimize import minimize_scalar,minimize
        # from scipy.optimize import brentq

        p = self.Pitch
        h = self.WedgeParameters['Height']

        cw = self.WedgeParameters['Velocity']

        cphi = np.cos(self.WedgeParameters['Angle'] * np.pi / 180.)
        sphi = np.sin(self.WedgeParameters['Angle'] * np.pi / 180.)

        # x,y = np.meshgrid(xrng,yrng)


        # def f(x,X,Y,n):
        #
        #     return np.sqrt((h + n * p * sphi)**2 + (cphi * n * p - x)**2) / cw + np.sqrt(Y**2 + (X - x)**2) / c
        # #
        # def J(x,X,Y,n):
        #     return -(cphi * n * p - x) / (cw * np.sqrt((h + n * p * sphi)**2 + (cphi * n * p - x)**2)) - \
        #             (X - x) / (c * np.sqrt(Y**2 + (X - x)**2))


        # def f(x,X,Y,n):
        #
        #     return (x - cphi*n*p)*(c * np.sqrt(Y**2 + (X - x)**2))/(cw * np.sqrt((h + n * p * sphi)**2 + (x - cphi*n*p)**2)) - (X - x)


        # self.Delays = [np.array([[minimize(f,x0=0.5 * np.abs(x - n * self.Pitch * cphi),
        #                 args=(x,y,n),method='BFGS',jac=J).fun for y in yrng] for x in xrng])
        #                for n in range(self.NumberOfElements)]


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


        x,y = np.meshgrid(xrng,yrng)

        ComputeDelays = np.vectorize(f,excluded=['n'])

        self.Delays = [ComputeDelays(x,y,n) for n in range(self.NumberOfElements)]


        # ComputeDelays = np.vectorize(f)

        # self.Delays = []
        #
        # for n in range(self.NumberOfElements):
        #
        #     ComputeDelays = np.vectorize(f)
        #
        #     self.Delays.append(ComputeDelays(x,y))






        # def ComputeDelay(n,x,y):
        #
        #     try:
        #         # return brentq(f,n*p*cphi, (x-n*p*cphi)*(h+n*p*sphi)/(y+h+n*p*sphi), args=(x,y,n),xtol=1e-6)
        #
        #         # return minimize_scalar(f,(n*p*cphi,(x-n*p*cphi)*(h+n*p*sphi)/(y+h+n*p*sphi)),args = (x,y,n),tol=1e-4,options={'maxiter':30}).x
        #         # return brentq(f,n*p*cphi, x, args=(x,y,n),xtol=-6)
        #
        #         return minimize(f,0.5*(n*p*cphi + (x-n*p*cphi)*(h+n*p*sphi)/(y+h+n*p*sphi)),args=(x,y,n),jac=J,tol=1e-4,options={'maxiter':20}).fun
        #
        #
        #     except ValueError:
        #
        #         return np.nan

        # self.Delays = [np.array([[brentq(f, n*p*cphi, (x-n*p*cphi)*(h+n*p*sphi)/(y+h+n*p*sphi), args=(x,y,n)) for x in xrng] for y in yrng]) for n in range(self.NumberOfElements)]

        # self.Delays = [np.array([[ComputeDelay(n,x,y) for y in yrng] for x in xrng]) for n in range(self.NumberOfElements)]



        self.xRange = xrng.copy()
        self.yRange = yrng.copy()

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

    def FitInterfaceCurve(self, ScanIndex, hrng, c, smoothparams=(0.1, 0.1)):

        from scipy.interpolate import interp1d
        # from scipy.signal import guassian, convolve
        from skimage.filters import threshold_li, gaussian
        from misc import DiffCentral

        xrng = np.linspace(
            0,
            self.NumberOfElements *
            self.Pitch,
            self.NumberOfElements)

        self.GetContactDelays(xrng, hrng, c)

        I = gaussian(np.abs(self.ApplyTFM(ScanIndex)), smoothparams)

        dh = hrng[1] - hrng[0]

        hgrid = hrng[0] + dh * np.argmax(I, axis=0)

        dhdx = np.abs(DiffCentral(hgrid))

        indkeep = np.where(dhdx < threshold_li(dhdx))

        hgrid = hgrid[indkeep]

        xrng = xrng[indkeep]

        # hgrid = np.zeros((options['NPeaks'], I.shape[1]))
        #
        # for i in range(I.shape[0]):
        #
        #     indmax,valmax = argrelmax(np.abs(I[:,i]), order=options['MinSpacing'])
        #
        #     indmax = indmax[np.argsort(valmax)[-options['NPeaks']::]]
        #
        #     hgrid[:,i] = hrng[0] + dh*indmax

        h = interp1d(
            xrng,
            hgrid,
            kind='quadratic',
            bounds_error=False,
            fill_value=np.nan)

        dhdx = interp1d(xrng[1::],
                        np.diff(h(xrng)) / np.diff(xrng),
                        bounds_error=False,
                        fill_value=np.nan)

        return h, dhdx

        # return h

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

    def GetAdaptiveDelays(self, xrng, yrng, h, cw, cs, AsParallel=False):

        # def GetAdaptiveDelays(self, xrng, yrng, h, dhdx, cw, cs,
        # AsParallel=False):

        from scipy.optimize import minimize_scalar
        # from scipy.interpolate import interp1d
        # from skimage.filters import threshold_li

        # xrng = np.linspace(0, self.NumberOfElements - 1, self.NumberOfElements) * self.Pitch

        # self.GetContactDelays(xrng, hrng, cw)
        #
        # I = self.ApplyTFM(ScanIndex,filterparams)
        #
        # dh = hrng[1] - hrng[0]
        #
        # hgrid = np.argmax(np.abs(I), axis=0) * dh + hrng[0]
        #
        # hthresh = threshold_li(hgrid)
        #
        # # hgrid = hgrid[hgrid>hthresh]
        #
        # h = interp1d(xrng, hgrid, bounds_error=False)

        def f(x, X, Y, n):
            return np.sqrt((x - n * self.Pitch)**2 + h(x) ** 2) / \
                cw + np.sqrt((X - x)**2 + (Y - h(x))**2) / cs

        # def dfdx(x, X, Y, n):
        #     return ((x - n * self.Pitch + h(x))**(-1 / 2.)) * (x - n * self.Pitch + h(x) * dhdx(x)) / \
        #         cw - (((X - x)**2 + (Y - h(x))**2)**(-1 / 2.)) * ((X - x)**2 + (Y - h(x)) * dhdx(x)) / cs

        # hmin = np.min(hgrid[hgrid>hthresh])
        #
        # yrng = np.linspace(hmin,hmin+depth[0],int(round(depth[0]/depth[1])))
        #
        # xrng = np.linspace(0,xrng[-1],int(round(xrng[-1]/depth[1])))

        # h0 = h(xrng[0])
        # mh = (h(xrng[-1]) - h0)/(xrng[-1]-xrng[0])
        #
        # def x0(X,Y,n):
        #
        #
        #     m = Y/(X-n*self.Pitch)
        #
        #     if np.isfinite(m):
        #
        #         return (m*n*self.Pitch - mh*xrng[0] + h0 - Y)/(m - mh)
        #
        #     else:
        #
        #         return X
        #

        # def DelayMin(n):
        #
        #     return [[float(minimize(f,x0=x0(x,y,n),args=(x,y,n),method='BFGS',options={'disp': False,
        #         'gtol': 1e-3,'eps': 1e-4,'return_all': False,'maxiter': 50, 'norm': np.inf}).fun)
        #         if y >= h(x) else np.nan for y in yrng] for x in xrng]

        def DelayMin(n):

            return [
                [
                    float(
                        minimize_scalar(
                            f,
                            bracket=(
                                n *
                                self.Pitch,
                                x),
                            args=(
                                x,
                                y,
                                n),
                            method='Brent',
                            options={
                                'xtol': 1e-3,
                                'maxiter': 50}).fun) if y >= h(x) else np.nan for y in yrng] for x in xrng]

        # def DelayMin(n):
        #
        #     return [[ float(dfdx(brentq(dfdx, n*self.Pitch,x,args=(x,y,n), maxiter=20, xtol=1e-3), x,y,n))
        #             if y >= h(x) else np.nan for y in yrng] for x in xrng]

        # self.Delays =
        # [[[float(minimize(f,x0=0.5*abs(x-n*self.Pitch),args=(x,y,n),method='BFGS',options={'disp':
        # False, 'gtol': 1e-3, 'eps': 1e-4, 'return_all': False, 'maxiter': 50,
        # 'norm': inf}).fun) if y>=h(x) else nan for y in yrng] for x in xrng]
        # for n in range(self.NumberOfElements)]

        if AsParallel:

            self.Delays = ProcessingPool().map(
                DelayMin, [n for n in range(self.NumberOfElements)])

        else:

            self.Delays = [DelayMin(n) for n in range(self.NumberOfElements)]

        self.xRange = xrng

        self.yRange = yrng

    def FilterByAngle(self, ScanIndex, filtertype, angle, FWHM, c):

        L = self.AScans[ScanIndex].shape[2]

        # Lpad = NextPow2(L)

        X = fftshift(
            fftn(
                np.real(
                    self.AScans[ScanIndex]), s=(
                    self.NumberOfElements, L), axes=(
                    0, 2)), axes=(0))

        # X = fftshift(rfft(fft(np.real(self.AScans[ScanIndex]), axis = 0)), axes = (0))

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

    def ApplyTFM(self, ScanIndex, FilterParams=None, stablecoeff=1e-4, Normalize=False):
        #
        # IX = len(self.Delays[0])
        # IY = len(self.Delays[0][0])

        if FilterParams is None:

            a = self.AScans[ScanIndex]

        else:

            a = self.FilterByAngle(
                ScanIndex,
                FilterParams[0],
                FilterParams[1],
                FilterParams[2],
                FilterParams[3])


        # L = self.AScans[ScanIndex].shape[2]
        #
        L = a.shape[2]

        t = np.linspace(0.,L-1,L)/self.SamplingFrequency
        #


        # delaytype = (len(self.Delays) == len(self.AScans))
        #
        # if self.AmplitudeCorrection is None:
        #
        #     def PointFocus(pt):
        #
        #     # Nd = len(self.Delays)
        #
        #         ix = pt[0]
        #         iy = pt[1]
        #
        #
        #
        #         I = 0.+0j
        #
        #         for m in range(Nd):
        #             for n in range(Nd):
        #
        #                 try:
        #
        #                     d = int(np.round((self.Delay[m][ix][iy]+self.Delay[n][ix][iy])*self.SamplingFrequency))
        #
        #                     I += a[m,n,int(d)]
        #
        #                 except:
        #
        #                     pass
        #
        #         return I
        #
        #
        #         # return reduce(lambda x, y: x +y, (a[m, n, int(np.round((self.Delays[m][ix][iy]+self.Delays[n][ix][iy]) * self.SamplingFrequency))] if (np.isfinite(self.Delays[m][ix][iy])\
        #         # and np.isfinite(self.Delays[n][ix][iy]) and int(round((self.Delays[m][ix][iy]+self.Delays[n][ix][iy]) * self.SamplingFrequency)) < L)\
        #         # else 0. +0j for n in range(Nd) for m in range(Nd)))
        #
        #
        # else:
        #
        #     def PointFocus(pt):
        #
        #         ix = pt[0]
        #         iy = pt[1]
        #
        #
        #         I = 0.+0j
        #
        #         for m in range(Nd):
        #             for n in range(Nd):
        #
        #                 try:
        #
        #                     d = int(np.round((self.Delay[m][ix][iy]+self.Delay[n][ix][iy])*self.SamplingFrequency))
        #
        #                     A = self.self.AmplitudeCorrection[m][iy,ix]*self.AmplitudeCorrection[n][iy,ix]
        #
        #                     if (not(np.isnan(A))):
        #
        #                         I += a[m,n,int(d)]/(A+stablecoeff)
        #
        #                 except:
        #
        #                     pass
        #
        #         return I
        #
        #         # return reduce(lambda x, y: x +y, (a[m, n, int(np.round((self.Delays[m][ix][iy]+self.Delays[n][ix][iy])*self.SamplingFrequency))]/(self.AmplitudeCorrection[m][iy,ix]*self.AmplitudeCorrection[n][iy,ix] + stablecoeff)
        #         # if (np.isfinite(self.Delays[m][ix][iy]) and np.isfinite(self.Delays[n][ix][iy]) and np.isfinite(self.AmplitudeCorrection[m][iy,ix]) and np.isfinite(self.AmplitudeCorrection[n][iy,ix])
        #         # and int(round((self.Delays[m][ix][iy]+self.Delays[n][ix][iy])*self.SamplingFrequency)) < L) else 0.+0j for n in range(Nd) for m in range(Nd)))
        #
        # if AsParallel:
        #
        #     pool_size = multiprocessing.cpu_count()
        #     os.system('taskset -cp 0-%d %s' % (pool_size, os.getpid()))
        #
        #     return np.array(
        #         ProcessingPool(pool_size).map(
        #             PointFocus, [
        #                 (ix, iy) for ix in range(IX) for iy in range(IY)])).reshape(
        #         (IX, IY)).transpose()
        #
        # else:
        #
        #     return np.array([PointFocus((ix, iy)) for ix in range(IX)
        #                      for iy in range(IY)]).reshape((IX, IY)).transpose()



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
