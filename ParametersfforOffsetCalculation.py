from sympy import *
import numpy as np

n = Symbol('n')
m = Symbol('m')
p = Symbol('p')
h = Symbol('h')
d = Symbol('d')
Th = Symbol('Th')
sphi = Symbol('sphi')
cphi = Symbol('cphi')

xn = Symbol('xn')
yn = Symbol('yn')
xn = n*p*cphi
yn = h + n*p*sphi

xm = Symbol('xm')
ym = Symbol('ym')
xm = d - m*p*cphi
ym = h + m*p*sphi

x0 = Symbol('x0')
x1 = Symbol('x1')
x2 = Symbol('x2')

cw = Symbol('cw')
c0 = Symbol('c0')
c1 = Symbol('c1')

t = Symbol('t')
dtdx0 = Symbol('dtdx0')
dtdx1 = Symbol('dtdx1')
dtdx2 = Symbol('dtdx2')

t = sqrt((xn-x0)**2+yn**2)/cw + sqrt((x1-x0)**2+Th**2)/c0 + sqrt((x2-x1)**2+Th**2)/c1 + sqrt((xm-x2)**2+ym**2)/cw
dtdx0 = simplify(diff(t,x0))
dtdx1 = simplify(diff(t,x1))
dtdx2 = simplify(diff(t,x2))






def EstimateProbesOffset(self,c,Offset,Thickness):

    from scipy.signal import hilbert
    from scipy.optimize import minimize_scalar,minimize
    from scipy.signal import argrelextrema

    p = self.Pitch
    N = self.NumberOfElements
    h = self.WedgeParameters['Height']
    phiw = self.WedgeParameters['Angle']*np.pi/180
    cw = self.WedgeParameters['Velocity']
    Fs = self.SamplingFrequency
    Scans = self.AScans[0]
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

    W = np.array([[int(np.round(0.15*T[n,m])) for m in range(int(N/2))] for n in range(int(N/2))])

    # MeasuredTimes = np.array([[np.argmax(np.abs(hilbert(np.real(Scans[n,m,T[n,m]-W[n,m]:T[n,m]+W[n,m]]))))+T[n,m]-W[n,m] for m in range(int(N/2))] for n in range(int(N/2))])
    MeasuredTimes = np.zeros((int(N/2),int(N/2)))

    for n in range(len(int(N/2))):

        for m in range(len(int(N/2))):

            A = np.abs(hilbert(np.real(Scans[n,m,T[n,m]-W[n,m]:T[n,m]+W[n,m]])))

            ind = argrelextrema(A,greater,order=4)

            if (len(ind[0])<1):
                MeasuredTimes[n,m] = np.nan

            else if ((len(ind[0])<2) & (len(ind[0])>0)):

                MeasuredTimes[n,m] = ind[0] + T[n,m] - W[n,m]

            else:

                Ind = np.argmin(np.abs(W[n,m] - ind[0]))
                MeasuredTimes[n,m] = ind[0][Ind] + T[n,m] - W[n,m]












    # error = np.abs(T-MeasuredTimes)
    #
    # mm = np.arange(16,32,2)
    #
    # X = []
    #
    # for i in range(len(mm)):
    #
    #     ind = np.argsort(error.flatten())[0:int(mm[i])]
    #
    #     Elements= np.unravel_index(np.array(ind),error.shape)
    #
    #     MT = [[MeasuredTimes[Elements[0][n],Elements[1][m]] for m in range(int(mm[i]))] for n in range(int(mm[i]))]
    #
    #     nn = int(mm[i])
    #
    #     res = minimize_scalar(Error,(Offset-5.,Offset+5.))
    #
    #     X.append(res.x)
    #
    # return X

    res = minimize_scalar(Error,(Offset-5.,Offset+5.))

    return res




# def EstimateProbesOffset(self,c,Offset,Thickness, Elements=None):
#
#     from scipy.signal import hilbert
#     from scipy.optimize import minimize_scalar,minimize
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
#     if Elements is None:
#
#         Elements = (list(range(int(N/2))),list(range(int(N/2),int(N))))
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
#
#     T = np.array([[int(np.round(Fs*delays(Elements[0][n],Elements[1][m]-int(N/2),Offset))) for m in range(len(Elements[1])))] for n in range(Elements[0])])
#
#     W = np.array([[int(np.round(0.15*T[n,m])) for m in range(len(Elements[1]))] for n in range(len(Elements[0]))])
#
#     MeasuredTimes = np.array([[np.argmax(np.abs(hilbert(np.real(Scans[Elements[0][n],Elements[1][m],T[n,m]-W[n,m]:T[n,m]+W[n,m]]))))+T[n,m]-W[n,m] for m in range(len(Elements[1]))] for n in range(Elements[0])])
#
#     return (T, MeasuredTimes, abs(T-MeasuredTimes))
#
#     TrElements = []
#     RcElements = []
#
#     for n in range(len(int(N/2))):
#         for m in range(len(int(N/2))):
#             if (np.abs(error[n,m]) < 15):
#                 TrElements.append(n)
#                 RCElements.append(m)
#     Elements = (TrElements,RcElements)
#
#
#     # def Error(d):
    #
    #     T = np.array([[int(np.round(Fs*delays(n,m,d))) for m in range(int(N/2))] for n in range(int(N/2))])
    #
    #     return np.sum(np.sum((MeasuredTimes - T)**2))
    #
    # res = minimize_scalar(Error,(Offset-5.,Offset+5.))
    #
    # return res
