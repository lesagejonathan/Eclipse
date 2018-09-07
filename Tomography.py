import numpy as np
import FMC
# import misc
import copy


def GetSpectralRange(X,Y,dB=-6):

    return np.where((np.abs(X)>=np.amax(np.abs(X))*10**(dB/20))&((np.abs(Y)>=np.amax(np.abs(Y))*10**(dB/20))))[0]


def FitPowerLaw(f,G,n):

    A = np.hstack(((f**n).reshape(-1,1), np.ones((len(f),1))))

    r = np.linalg.lstsq(A,G.reshape(-1,1))[0]


    return r[0],r[1],np.trapz(G/(f[-1]-f[0]),dx=f[1]-f[0])


def ToHalfMatrix(a):

    for i in range(a.shape[-1]):

        a[:,:,i] = 0.5*(np.tril(a[:,:,i]).transpose() + np.triu(a[:,:,i]))

    return a

# def BilinearInterp(x,y,xi,yi,fi):
#
#     from numpy.linalg import solve
#
#
#     def Interp(x,y):
#
#         indx = np.argmin(np.abs(x-xi))
#         indy = np.argmin(np.abs(y-yi))
#
#         if indx == 0:
#
#             ix1 = indx
#             ix2 = indx+1
#
#         elif indx == len(xi)-1:
#
#             indx1 = indx-1
#             indx2 = indx
#
#         else:
#
#             Xim1 = x[indx-1]
#             Xi = x[indx]
#             Xip1 = x[indx + 1]
#
#             if (0.5*(Xim1+Xi))<=(0.5*(Xi+Xip1)):
#
#                 indx1 = indx
#                 indx2 = indx + 1
#
#             else:
#
#                 indx1 = indx - 1
#                 indx2 = indx
#
#
#         if indy == 0:
#
#             indy1 = indy
#             indy2 = indy+1
#
#         elif indy == len(yi)-1:
#
#             indy1 = indy-1
#             indy2 = indy
#
#         else:
#
#             Yim1 = y[indy-1]
#             Yi = y[indy]
#             Yip1 = y[indy + 1]
#
#             if (0.5*(Yim1+Yi))<=(0.5*(Yi+Yip1)):
#
#                 indy1 = indy
#                 indy2 = indy+1
#
#             else:
#
#                 indy1 = indy-1
#                 indy2 = indy
#
#         x1 = xi[indx1]
#         x2 = xi[indx2]
#         y1 = yi[indy1]
#         y2 = yi[indy2]
#
#         f = np.array([fi[indx1,indy1],fi[indx1,indy2],fi[indx2,indy1],fi[indx2,indy2]]).reshape(-1,1)
#
#         A = np.array([[1.,x1,y1,x1*y1],[1.,x1,y2,x1*y2],[1.,x2,y1,x2*y1],[1.,x2,y2,x2*y2]])
#
#         a = solve(A,f)
#
#         return a[0] + a[1]*x + a[2]*y + a[3]*x*y
#
#     I = np.vectorize(Interp)
#
#     return I(x,y)


def BilinearInterp(x,y,xi,yi,fi):

    from numpy.linalg import solve

    def Interp(x,y):

        if (x<np.amin(xi))&(x>np.amax(xi))&(y<np.amin(yi))&(y>np.amax(yi)):

            I = 0.

        else:


            r = np.sqrt((x-xi)**2 + (y-yi)**2)

            ind = np.argsort(r)[0:3]

            x0,y0 = xi[ind[0]],yi[ind[0]]
            x1,y1 = xi[ind[1]],yi[ind[1]]
            x2,y2 = xi[ind[2]],yi[ind[2]]
            x3,y3 = xi[ind[3]],yi[ind[3]]

            f = np.array([fi[ind[0]],fi[ind[1]],fi[ind[2]],fi[ind[3]]]).reshape(-1,1)

            A = np.array([[1.,x0,y0,x0*y0],[1.,x1,y1,x1*y1],[1.,x2,y2,x2*y2],[1.,x3,y3,x3*y3]])

            I = np.dot(np.array([1.,x,y,x*y]).reshape(1,-1),solve(A,f))

        return I

    I = np.vectorize(x,y)

    return I


def BilinearInterpCoeffs(x,y,xi,yi):

    from numpy.linalg import solve,inv,det

    r = np.sqrt((x-xi)**2 + (y-yi)**2)

    ind = np.argsort(r)[0:4]

    x0,y0 = xi[ind[0]],yi[ind[0]]
    x1,y1 = xi[ind[1]],yi[ind[1]]
    x2,y2 = xi[ind[2]],yi[ind[2]]
    x3,y3 = xi[ind[3]],yi[ind[3]]

    A = np.array([[1.,x0,y0,x0*y0],[1.,x1,y1,x1*y1],[1.,x2,y2,x2*y2],[1.,x3,y3,x3*y3]])

    coeffs = np.dot(np.array([1.,x,y,x*y]).reshape(1,-1),inv(A)).flatten()

    return ind, coeffs


class TomographyCapture:

    def __init__(self, capture):


        self.Capture = copy.deepcopy(capture)

        self.PathParameters = {}

        self.ProjectionMatrixInverse = {}

        self.ProjectionMatrix = {}


    def ToHalfMatrixCapture(self):

        self.Capture.AScans = [ToHalfMatrix(a) for a in self.Capture.AScans]


    def GetPathParameters(self,pathkey,params):

        """ Sets PathParameters attribute with dictionary having intersections with geometry, delays between element pairs and
        path lengths inside inspection geometry. Used for computing projection matrix """

        self.PathParameters[pathkey] = {}

        N = self.Capture.NumberOfElements

        p = self.Capture.Pitch

        x = np.linspace(-0.5*(N-1)*p, 0.5*(N-1)*p, N)

        if pathkey== 'Contact':

            xymn = [[((x[m],0.),(0.5*(x[n]-x[m]),params['Thickness']),(x[n],0.)) for m in range(N)] for n in range(N)]

            lmn = [[ (np.sqrt((xymn[m][n][1][0] - xymn[m][n][0][0])**2 + (xymn[m][n][1][1] - xymn[m][n][1][0])**2),np.sqrt((xymn[m][n][2][0] - xymn[m][n][1][0])**2 + (xymn[m][n][2][1] - xymn[m][n][1][0])**2)) for m in range(N)] for n in range(N)]

            self.PathParameters[pathkey]['Delays'] = [[lmn[m][n][0]/params['Velocity'] + lmn[m][n][1]/params['Velocity'] for m in range(N)] for n in range(N)]

            self.PathParameters[pathkey]['Lengths'] = lmn

            self.PathParameters[pathkey]['Intersections'] = xymn

            self.PathParameters[pathkey]['Thickness'] = params['Thickness']




    def GetProjectionMatrix(self,pathkey,ds=0.1,svdcutoff=None):

        """
            Computes projection matrix if svdcutoff is not None GetProjectionMatrixInverse is called with svdcutoff value
        """

        from numpy.linalg import pinv, svd

        N = self.Capture.NumberOfElements

        M = int(np.round((1+N)*N/2))


        B = []

        for m in range(N):

            for n in range(m,N):

                BB = np.zeros((M,1))

                lab = self.PathParameters[pathkey]['Lengths'][m][n][0]

                lbc = self.PathParameters[pathkey]['Lengths'][m][n][1]

                L = lab + lbc

                s = np.linspace(0,L,int(np.round(L/ds)))

                K = len(s)

                xa = self.PathParameters[pathkey]['Intersections'][m][n][0][0]
                xb = self.PathParameters[pathkey]['Intersections'][m][n][1][0]
                xc = self.PathParameters[pathkey]['Intersections'][m][n][2][0]

                ya = self.PathParameters[pathkey]['Intersections'][m][n][0][1]
                yb = self.PathParameters[pathkey]['Intersections'][m][n][1][1]
                yc = self.PathParameters[pathkey]['Intersections'][m][n][2][1]

                for k in range(K):

                    sk = s[k]

                    if sk<=lab:

                        xk = xa + sk*(xb-xa)/lab
                        yk = ya + sk*(yb-ya)/lab

                    elif (sk>lab)&(sk<=L):

                        xk = xb + (sk-lab)*(xc-xb)/lbc
                        yk = yb + (sk-lab)*(yc-yb)/lbc


                    ind,c = BilinearInterpCoeffs(xk,yk,self.Grid['x'],self.Grid['y'])

                    for i in range(4):

                        BB[ind[i]] = BB[ind[i]] + c[i]

                B.append(BB)

        B = np.array(B)

        self.ProjectionMatrix[pathkey] = B

        if svdcutoff is not None:

            self.GetProjectionMatrixInverse(pathkey,svdcutoff)




    def GetProjectionMatrixInverse(self,pathkey,svdcutoff=0.99):

        from numpy.linalg import svd, pinv

        B = self.ProjectionMatrix[pathkey]

        ev = svd(B)[1]

        ev = ev/np.amax(ev)

        e = ev/np.sum(ev)

        e = np.array([np.sum(e[0:i]) for i in range(len(e))])

        ind = np.argmin(np.abs(e-svdcutoff))

        self.ProjectionMatrixInverse[pathkey] = pinv(B,ev[ind])


    def SetGrid(self,pathkey):

        if pathkey=='Contact':

            p = self.Capture.Pitch
            N = self.Capture.NumberOfElements

            M = int(np.round((1+N)*N/2))

            Ny = int(np.round(M/N))

            x = np.linspace(-p*(N-1)*0.5,p*(N-1)*0.5,N)
            y = np.linspace(0.,self.PathParameters[pathkey]['Thickness'],Ny)

            xy = np.meshgrid(x,y)

            self.Grid = {'x':xy[0].flatten(), 'y':xy[1].flatten()}


    def GetAttenuationImage(self, pathkey, ScanIndex, RefIndex, fpower, resolution=0.1, fband=None, windowparams=(50, 0.1), fftpad = 4):


        from numpy.fft import ifft,fftshift,rfft,ifftshift
        from scipy.signal import tukey
        from scipy.ndimage import zoom
        from matplotlib.pyplot import plot,show
        from scipy.sparse.linalg import lsqr
        # from numpy.linalg import dot


        fs = self.Capture.SamplingFrequency

        Gavg = []
        Gexp = []


        W = tukey(int(2*windowparams[0]),windowparams[1])

        NFFT = int(fftpad*2*windowparams[0])


        a = self.Capture.AScans[ScanIndex]
        aref = self.Capture.AScans[RefIndex]

        f = np.linspace(0., fs/2, np.floor(NFFT/2) + 1)
        N = self.Capture.NumberOfElements


        for m in range(N):

            for n in range(m,N):


                ind = int(np.round(fs*self.PathParameters[pathkey]['Delays'][m][n]))

                Aref = rfft(W*aref[m,n,ind-windowparams[0]:ind+windowparams[0]], NFFT)

                A = rfft(W*a[m,n,ind-windowparams[0]:ind+windowparams[0]], NFFT)


                if fband is None:

                    indf = GetSpectralRange(Aref,A)

                else:

                    indf = np.where((f>=fband[0])&(f<=fband[1]))[0]


                Aref = Aref[indf]
                A = A[indf]

                Arefmax = np.amax(abs(Aref))

                v = FitPowerLaw(f[indf],-np.log(np.abs(A)/np.abs(Aref)),fpower)

                Gavg.append(v[2])
                Gexp.append(v[0])



        Gexp = np.array(Gexp).reshape(-1,1)
        Gavg = np.array(Gavg).reshape(-1,1)


        Iexp = np.dot(self.ProjectionMatrixInverse[pathkey],Gexp)
        Iavg = np.dot(self.ProjectionMatrixInverse[pathkey],Gavg)


        xmin = np.amin(self.Grid['x'])
        xmax = np.amax(self.Grid['x'])

        ymin = np.amin(self.Grid['y'])
        ymax = np.amax(self.Grid['y'])

        x,y = np.meshgrid(np.linspace(xmin,xmax,int(np.round((xmax-xmin)/resolution))), np.linspace(ymin,ymax,int(np.round((ymax-ymin)/resolution))))

        Iexp = BilinearInterp(x,y,self.Grid['x'],self.Grid['y'],Iexp)

        Iavg = BilinearInterp(x,y,self.Grid['x'],self.Grid['y'],Iavg)


        return Iavg,Iexp












































    # def GetInverse(self,svdcutoff=0.99,ds=0.1):
    #
    #     from numpy.fft import ifft,fftshift,rfft,ifftshift,ifftn
    #     from scipy.signal import tukey
    #
    #     N = self.Capture.NumberOfElements
    #
    #     M = int(np.round((1+N)*N/2))
    #
    #     # M = N**2
    #
    #
    #     # c = self.Capture.WaveSpeed
    #
    #     p = self.Capture.Pitch
    #
    #     dx = p
    #
    #     dy = p
    #
    #     Nx = int(np.round(((N-1)*p)/dx))
    #
    #     # Ny = int(round(M/Nx))
    #
    #     Ny = int(d/(2*dy))
    #
    #
    #     # dy = d/Ny
    #
    #     # dy = d/Ny
    #     #
    #     # print(dy)
    #     # print(Ny)
    #
    #     # Ny = int(np.round(d/dy))
    #     # Nx = int(np.round(((N-1)*p)/dx))
    #
    #
    #     aref = ToHalfMatrix(self.Capture.AScans[RefIndex])
    #     a = ToHalfMatrix(self.Capture.AScans[ScanIndex])
    #
    #
    #     # aref = self.Capture.AScans[RefIndex]
    #     # a = self.Capture.AScans[ScanIndex]
    #
    #     Kx,Ky = np.meshgrid(2*np.pi*np.linspace(-1/(2*dx),1/(2*dx), Nx) + 0j, 2*np.pi*np.linspace(0,1/(2*dy), Ny) + 0j)
    #
    #     kx = Kx.flatten().astype(np.complex128)
    #     ky = Ky.flatten().astype(np.complex128)
    #
    #
    #
    #     # B = np.zeros((M,len(kx)),dtype=np.complex128)
    #
    #     x = np.linspace(-0.5*p*(N-1),0.5*p*(N-1),N)
    #
    #     # x = np.linspace(0,p*(N-1),N)
    #
    #
    #
    #     for m in range(N):
    #
    #         for n in range(m,N):
    #
    #
    #             BB = zeros(())
    #
    #
    #             for k in range(len()):
    #
    #                 # BB[] += BB[]
    #
    #
    #
    #         # for n in range(N):
    #
    #
    #
    #             ind = int(np.round(self.Capture.SamplingFrequency*np.sqrt((x[n]-x[m])**2 + (2*d)**2)/c))
    #
    #             Aref = rfft(W*aref[m,n,ind-windowparams[0]:ind+windowparams[0]], NFFT)
    #
    #             A = rfft(W*a[m,n,ind-windowparams[0]:ind+windowparams[0]], NFFT)
    #
    #
    #             indfreq = GetSpectralRange(Aref,A)
    #
    #             # print(A.shape)
    #
    #             v = FitPowerLaw(f[indfreq], np.log(np.abs(Aref)/np.abs(A))[indfreq],fpower)
    #
    #             Gexp.append(v[0])
    #             Gavg.append(v[2])
    #
    #             B.append(Integral(m,n))
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # def GetGridImage(self, ScanIndex, RefIndex, d, fpower, c, resolution=0.1, windowparams=(50, 0.1, 4), rcondnum = 1e-2):
    #
    #
    #     from numpy.fft import ifft,fftshift,rfft,ifftshift
    #     from scipy.signal import tukey
    #     from scipy.ndimage import zoom
    #     from matplotlib.pyplot import plot,show
    #     from scipy.sparse.linalg import lsqr
    #
    #     N = self.Capture.NumberOfElements
    #
    #     # c = self.Capture.WaveSpeed
    #
    #     p = self.Capture.Pitch
    #
    #     M = int(np.round((1+N)*N/2))
    #
    #     # dx = np.sqrt(((N-1)*p*d)/M)
    #
    #     # dx = p
    #     #
    #     # dy = p
    #     #
    #     dx = resolution
    #     dy = resolution
    #
    #     # dx = np.sqrt(d*p*(N-1)/M)
    #     # dy = p*(N-1)*d/(dx*M)
    #
    #
    #
    #     # Nx = int(np.round(((N-1)*p)/dx))
    #     #
    #     # # Ny = int(M/Nx)
    #     # #
    #     # #
    #     # # dy = d/Ny
    #     #
    #     # Ny = int(np.round(d/dy))
    #
    #
    #     # dx = 2*p
    #
    #
    #     # dx = self.Capture.Pitch
    #     # dy = dx
    #
    #     # Nx = int(np.round(((N-1)*p)/dx))
    #     #
    #     # Ny = int(M/Nx)
    #     #
    #     # dy = d/Ny
    #
    #     #
    #     aref = ToHalfMatrix(self.Capture.AScans[RefIndex])
    #     a = ToHalfMatrix(self.Capture.AScans[ScanIndex])
    #
    #     # aref = self.Capture.AScans[RefIndex]
    #     # a = self.Capture.AScans[ScanIndex]
    #
    #
    #     fs = self.Capture.SamplingFrequency
    #
    #
    #
    #     # M = int(np.round((1+N)*N/2))
    #
    #
    #     Gavg = []
    #     Gexp = []
    #
    #     # B = []
    #
    #     W = tukey(int(2*windowparams[0]),windowparams[1])
    #
    #     NFFT = int(windowparams[2]*2*windowparams[0])
    #
    #     # B = np.zeros((int(N*N),int(Nx*Ny)),dtype=np.complex)
    #
    #
    #     f = np.linspace(0., fs/2, np.floor(NFFT/2) + 1)
    #
    #     x = np.linspace(-0.5*(N-1)*p,0.5*(N-1)*p,N)
    #
    #     # x = np.linspace(0.,(N-1)*p,N)
    #
    #     xgrid, ygrid = np.meshgrid(np.arange(x[0]+dx/2,x[-1],dx),np.arange(dy/2,d,dy))
    #
    #
    #     # xgrid, ygrid = np.meshgrid(np.arange(x[0],x[-1],dx),np.arange(0.,d,dy))
    #     #
    #     Nx = xgrid.shape[1]
    #     Ny = xgrid.shape[0]
    #     #
    #     #
    #     # xgrid = xgrid.reshape(1,-1)
    #     #
    #     # ygrid = ygrid.reshape(1,-1)
    #     #
    #     #
    #     # B = np.zeros((M,xgrid.shape[1]))
    #
    #     # B = np.zeros((M,Nx*Ny))
    #
    #
    #
    #
    #     B = []
    #
    #
    #     # print(B.shape)
    #
    #
    #
    #     def xray(y,m,n,pathind):
    #
    #
    #         if pathind == 0:
    #
    #             c0 = (x[n] - x[m])/(2*d)
    #
    #             c1 = x[m]
    #
    #
    #         elif pathind == 1:
    #
    #
    #             c0 = -(x[n] - x[m])/(2*d)
    #
    #             c1 = x[n]
    #
    #         return c0*y + c1
    #
    #
    #
    #     for m in range(N):
    #
    #         xm = x[m]
    #
    #         # for n in range(m,N):
    #
    #         # for n in range(N):
    #
    #         # BB = np.zeros((1,int(Nx*Ny)))
    #
    #
    #         for n in range(m,N):
    #
    #
    #             xn = x[n]
    #
    #             BB = np.zeros((1,int(Nx*Ny)))
    #
    #
    #
    #
    #             ind = int(np.round(fs*np.sqrt((xn-xm)**2 + (2*d)**2)/c))
    #
    #             Aref = rfft(W*aref[m,n,ind-windowparams[0]:ind+windowparams[0]], NFFT)
    #
    #             A = rfft(W*a[m,n,ind-windowparams[0]:ind+windowparams[0]], NFFT)
    #
    #
    #             indf = GetSpectralRange(Aref,A)
    #
    #
    #             Aref = Aref[indf]
    #             A = A[indf]
    #
    #             Arefmax = np.amax(abs(Aref))
    #
    #             v = FitPowerLaw(f[indf],np.log(np.abs(Aref)/np.abs(A)),fpower)
    #
    #             Gavg.append(v[2])
    #             Gexp.append(v[0])
    #
    #             B.append((np.abs(xray(ygrid,m,n,0)-xgrid)<dx).astype(np.float) + (np.abs(xray(ygrid,m,n,1)-xgrid)<dx).astype(np.float))
    #
    #             # BB = BB + (np.abs(xray(ygrid,m,n,0)-xgrid)<dx).astype(np.float) + (np.abs(xray(ygrid,m,n,1)-xgrid)<dx).astype(np.float)
    #
    #
    #             # B[n+m*(N-m),:] = B[n+m*(N-m),:] + (np.abs(xray(ygrid,m,n,0)-xgrid)<=dx).astype(np.float) + (np.abs(xray(ygrid,m,n,1)-xgrid)<=dx).astype(np.float)
    #
    #             # G[n+m*self.NumberOfElements, 0] = np.log(RR/R)
    #
    #
    #             # G = np.log(np.abs(R * np.conj(RR) /
    #             #                   (RR * np.conj(RR) + 1e-2 * np.amax(np.abs(RR)))))
    #
    #             # pfit = np.polyfit(f, G, 1)
    #
    #             # G0[n + m * self.NumberOfElements, 0] = pfit[1]
    #             # G1[n + m * self.NumberOfElements, 0] = -pfit[0]
    #
    #
    #             # G0[n+m*self.NumberOfElements, 0] = np.trapz(-G,dx=(f[1]-f[0]))/(fband[1]-fband[0])
    #             #
    #             # G1[n+m*self.NumberOfElements, 0] = np.trapz(-G/f,dx=(f[1]-f[0]))/(fband[1]-fband[0])
    #             #
    #             # GG.append(G0)
    #
    #
    #             # def xf(y):
    #             #     return (((xn-xm)*y)/(2*d)) + xm
    #             #
    #             # def yf(x):
    #             #     return ((x - xm)*2*d)/(xn - xm)
    #             #
    #             # def xb(y):
    #             #     return (((xm-xn)*y)/(2*d)) + xn
    #             #
    #             # def yb(x):
    #             #     return ((x - xn)*2*d)/(xm - xn)
    #             #
    #             #
    #             # xy = []
    #             #
    #             #
    #             # for p in range(int(round(np.ceil(xm/dx))), int(round(np.floor(xf(d)/dx)))):
    #             #
    #             #     xp = dx*p
    #             #
    #             #     yp = yf(xp)
    #             #
    #             #
    #             #     xy.append([xp, yp])
    #             #
    #             # for q in range(Ny):
    #             #
    #             #     yq = dy*q
    #             #
    #             #     xq = xf(yq)
    #             #
    #             #
    #             #     xy.append([xq,yq])
    #             #
    #             #
    #             # xy = np.array(xy).reshape((len(xy),2))
    #             #
    #             # ixysort = np.argsort(xy[:,1])
    #             #
    #             # xy = xy[ixysort,:]
    #             #
    #             #
    #             # for i in range(xy.shape[0]-1):
    #             #
    #             #     x0 = xy[i, 0]
    #             #     x1 = xy[i+1 , 0]
    #             #
    #             #     y0 = xy[i, 1]
    #             #     y1 = xy[i+1, 1]
    #             #
    #             #     p = int(np.floor(x0/dx))
    #             #     q = int(np.floor(y0/dy))
    #             #
    #             #     r = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    #             #     #
    #             #     # print(n + m*(N-m))
    #             #     #
    #             #     # print(p+q*Nx)
    #             #
    #             #     # B[n + m*N, p+q*Nx] = B[n + m*N, p+q*Nx] + r
    #             #
    #             #     BB[0,p+q*Nx] = BB[0,p+q*Nx] + r
    #             #
    #             #     # B[n + m*(N-m), p+q*Nx] = B[n + m*(N-m), p+q*Nx] + r
    #             #
    #             #
    #             # xy = []
    #             #
    #             #
    #             # for p in range(int(np.ceil(xf(d)/dx)), int(np.floor(xn/dx))):
    #             #
    #             #     xp = dx*p
    #             #
    #             #     yp = yb(xp)
    #             #
    #             #     xy.append([xp, yp])
    #             #
    #             #
    #             # for q in range(Ny):
    #             #
    #             #     yq = dy*q
    #             #
    #             #     xq = xb(yq)
    #             #
    #             #     xy.append([xq,yq])
    #             #
    #             #
    #             # xy = np.array(xy).reshape((len(xy),2))
    #             #
    #             # ixysort = np.argsort(xy[:,1])
    #             #
    #             # ixysort = ixysort[::-1]
    #             #
    #             # xy = xy[ixysort, :]
    #             #
    #             # for i in range(xy.shape[0]-1):
    #             #
    #             #     x0 = xy[i, 0]
    #             #     x1 = xy[i+1 , 0]
    #             #
    #             #     y0 = xy[i, 1]
    #             #     y1 = xy[i+1, 1]
    #             #
    #             #     p = int(np.floor(x0/dx))
    #             #     q = int(np.floor(y0/dy))
    #             #
    #             #     r = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    #             #
    #             #
    #             #     # B[n + m*(N-m), p+q*Nx] = B[n + m*(N-m), p+q*Nx] + r
    #             #
    #             #     BB[0,p+q*Nx] = BB[0,p+q*Nx] + r
    #             #
    #             # B.append(BB)
    #
    #
    #     B = np.array(B).reshape(M,int(Nx*Ny))
    #
    #
    #     # B = np.array(B).reshape((M,int(Nx*Ny)))
    #
    #     Gexp = np.array(Gexp).reshape(-1,1)
    #     Gavg = np.array(Gavg).reshape(-1,1)
    #
    #     # np.nan_to_num(Gexp,copy=False)
    #     # np.nan_to_num(Gavg,copy=False)
    #
    #     #
    #     Iexp = np.linalg.lstsq(B, Gexp, rcond=rcondnum)[0].reshape((Ny,Nx))
    #
    #     Iavg = np.linalg.lstsq(B, Gavg, rcond=rcondnum)[0].reshape((Ny,Nx))
    #
    #
    #     # Iexp = lsqr(B, Gexp)[0].reshape((Ny,Nx))
    #     #
    #     # Iavg = lsqr(B, Gavg)[0].reshape((Ny,Nx))
    #
    #
    #     return Iavg,Iexp,B
    #
    #
    #     # return zoom(Iexp,(dy/resolution,dx/resolution)), zoom(Iavg,(dy/resolution,dx/resolution)), B