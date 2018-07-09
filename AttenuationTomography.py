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


class AttenuationTomography:

    def __init__(self, capture):


        self.Capture = copy.deepcopy(capture)



    def GetSpectralImage(self, ScanIndex, RefIndex, d, fpower, c, resolution=0.1, windowparams=(50, 0.1, 4), rcondnum = 1e-2):

        from numpy.fft import ifft,fftshift,rfft,ifftshift
        from scipy.signal import tukey

        N = self.Capture.NumberOfElements

        # c = self.Capture.WaveSpeed

        p = self.Capture.Pitch

        dx = self.Capture.Pitch
        dy = dx

        Ny = int(np.round(d/dy))
        Nx = int(np.round(((N-1)*p)/dx))


        aref = ToHalfMatrix(self.Capture.AScans[RefIndex])
        a = ToHalfMatrix(self.Capture.AScans[ScanIndex])

        Kx,Ky = np.meshgrid(2*np.pi*np.linspace(-1/(2*dx),1/(2*dx), Nx) + 0j, 2*np.pi*np.linspace(0,1/(2*dy), Ny) + 0j)

        kx = Kx.flatten()
        ky = Ky.flatten()


        M = int(np.round((1+N)*N/2))

        # B = np.zeros((M,len(kx)),dtype=np.complex128)

        x = np.linspace(-0.5*p*(N-1),0.5*p*(N-1),N)

        # x = np.linspace(0,p*(N-1),N)


        def Integral(m,n,fb):


            if fb == 0:

                c0 = (x[n] - 3*x[m])/(2*d)

                c1 = x[m]


            elif fb == 1:

                # c0 = -(x[n]+x[m])/(2*d)

                c0 = (x[m]-x[n])/(2*d)

                c1 = x[n]



            I = np.exp(1j*kx*c1)*(np.exp(1j*d*(kx*c0+ky)) - 1)/(1j*(kx*c0+ky))

            infmask = np.logical_not(np.isfinite(I))

            I[infmask] = np.exp(1j*kx[infmask]*c1)*d

            return I.reshape(1,-1)

        Gavg = []
        Gexp = []

        B = []

        W = tukey(int(2*windowparams[0]),windowparams[1])

        NFFT = int(windowparams[2]*2*windowparams[0])

        f = np.linspace(0., self.Capture.SamplingFrequency/2., np.floor(NFFT/2) + 1)

        for m in range(N):

            for n in range(m,N):


                ind = int(np.round(self.Capture.SamplingFrequency*np.sqrt((x[n]-x[m])**2 + (2*d)**2)/c))

                Aref = rfft(W*aref[m,n,ind-windowparams[0]:ind+windowparams[0]], NFFT)

                A = rfft(W*a[m,n,ind-windowparams[0]:ind+windowparams[0]], NFFT)


                indfreq = GetSpectralRange(Aref,A)

                # print(A.shape)


                v = FitPowerLaw(f[indfreq], np.log(np.abs(Aref)/np.abs(A))[indfreq],fpower)

                Gexp.append(v[0])
                Gavg.append(v[2])

                B.append(Integral(m,n,0) - Integral(m,n,1))




        B = np.array(B).reshape((M,len(kx)))

        Gavg = np.array(Gavg,dtype=np.complex).reshape(-1,1)

        Gexp = np.array(Gexp,dtype=np.complex).reshape(-1,1)

        Aavg = np.linalg.lstsq(B,Gavg,rcondnum)[0].reshape(Kx.shape)

        Aexp = np.linalg.lstsq(B,Gexp,rcondnum)[0].reshape(Kx.shape)


        # Binv = np.linalg.pinv(B,rcondnum)
        #
        # Aavg = np.dot(Binv,Gavg).reshape(Kx.shape)
        #
        # Aexp = np.dot(Binv,Gexp).reshape(Ky.shape)


        Nx = int(np.round((N-1)*p/resolution))

        xzeropad = np.zeros((Aavg.shape[0],int(np.round((Nx - Aavg.shape[1]-2)/2))), dtype=complex)

        Aavg = ifft(ifftshift(np.hstack((xzeropad, Aavg, xzeropad)), axes=(1,)), axis=1)

        Aexp = ifft(ifftshift(np.hstack((xzeropad, Aexp, xzeropad)), axes=(1,)), axis=1)


        Ny = int(np.round(d/(2*resolution)))

        return fftshift(ifft(Aavg,axis=0,n=2*Ny - 2),axes=(1,)), fftshift(ifft(Aexp,axis=0,n=2*Ny - 2),axes=(1,)),Gexp,Gavg



    def GetGridImage(self, ScanIndex, RefIndex, d, fpower, c, resolution=0.1, windowparams=(50, 0.1, 4), rcondnum = 1e-2):


        from numpy.fft import ifft,fftshift,rfft,ifftshift
        from scipy.signal import tukey
        from scipy.ndimage import zoom
        from matplotlib.pyplot import plot,show

        N = self.Capture.NumberOfElements

        # c = self.Capture.WaveSpeed

        p = self.Capture.Pitch

        M = int(np.round((1+N)*N/2))



        dx = 2*p


        # dx = self.Capture.Pitch
        # dy = dx

        Nx = int(np.round(((N-1)*p)/dx))

        Ny = int(M/Nx)

        dy = d/Ny

        #
        aref = ToHalfMatrix(self.Capture.AScans[RefIndex])
        a = ToHalfMatrix(self.Capture.AScans[ScanIndex])

        # aref = self.Capture.AScans[RefIndex]
        # a = self.Capture.AScans[ScanIndex]


        fs = self.Capture.SamplingFrequency



        # M = int(np.round((1+N)*N/2))


        Gavg = []
        Gexp = []

        B = []

        W = tukey(int(2*windowparams[0]),windowparams[1])

        NFFT = int(windowparams[2]*2*windowparams[0])

        # B = np.zeros((int(N*N),int(Nx*Ny)),dtype=np.complex)


        f = np.linspace(0., fs/2, np.floor(NFFT/2) + 1)

        # x = np.linspace(-0.5*(N-1)*p,0.5*(N-1)*p,N)

        x = np.linspace(0.,(N-1)*p,N)

        # xgrid, ygrid = np.meshgrid(np.linspace(0.,(Nx-1)*dx,Nx),np.linspace(0.,(Ny-1)*dy,Ny))
        #
        # xgrid = xgrid.reshape(1,-1)
        #
        # ygrid = ygrid.reshape(1,-1)
        #
        #
        # B = np.zeros((int(N**2),xgrid.shape[1]))



        def xray(y,m,n,pathind):


            if pathind == 0:

                c0 = (x[n] - 3*x[m])/(2*d)

                c1 = x[m]


            elif pathind == 1:


                c0 = (x[m]-x[n])/(2*d)

                c1 = x[n]

            return c0*y + c1



        for m in range(N):

            xm = x[m]

            # for n in range(m,N):

            # for n in range(N):

            # BB = np.zeros((1,int(Nx*Ny)))


            for n in range(m,N):


                xn = x[n]



                ind = int(np.round(fs*np.sqrt((xn-xm)**2 + (2*d)**2)/c))

                Aref = rfft(W*aref[m,n,ind-windowparams[0]:ind+windowparams[0]], NFFT)

                A = rfft(W*a[m,n,ind-windowparams[0]:ind+windowparams[0]], NFFT)


                indf = GetSpectralRange(Aref,A)


                Aref = Aref[indf]
                A = A[indf]

                Arefmax = np.amax(abs(Aref))




                v = FitPowerLaw(f[indf],np.log(np.abs(Aref)/np.abs(A)),fpower)

                Gavg.append(v[2])
                Gexp.append(v[0])


                # BB = BB + (np.abs(xray(ygrid,m,n,0)-xgrid)<dx).astype(np.float) + (np.abs(xray(ygrid,m,n,1)-xgrid)<dx).astype(np.float)


                # B[n+m*N,:] = B[n+m*N,:] + (np.abs(xray(ygrid,m,n,0)-xgrid)<=dx).astype(np.float) + (np.abs(xray(ygrid,m,n,1)-xgrid)<=dx).astype(np.float)


                # G[n+m*self.NumberOfElements, 0] = np.log(RR/R)


                # G = np.log(np.abs(R * np.conj(RR) /
                #                   (RR * np.conj(RR) + 1e-2 * np.amax(np.abs(RR)))))

                # pfit = np.polyfit(f, G, 1)

                # G0[n + m * self.NumberOfElements, 0] = pfit[1]
                # G1[n + m * self.NumberOfElements, 0] = -pfit[0]


                # G0[n+m*self.NumberOfElements, 0] = np.trapz(-G,dx=(f[1]-f[0]))/(fband[1]-fband[0])
                #
                # G1[n+m*self.NumberOfElements, 0] = np.trapz(-G/f,dx=(f[1]-f[0]))/(fband[1]-fband[0])
                #
                # GG.append(G0)

                #
                def xf(y):
                    return (((xn-xm)*y)/(2*d)) + xm

                def yf(x):
                    return ((x - xm)*2*d)/(xn - xm)

                def xb(y):
                    return (((xm-xn)*y)/(2*d)) + xn

                def yb(x):
                    return ((x - xn)*2*d)/(xm - xn)


                xy = []


                for p in range(int(round(np.ceil(xm/dx))), int(round(np.floor(xf(d)/dx)))):

                    xp = dx*p

                    yp = yf(xp)


                    xy.append([xp, yp])

                for q in range(Ny):

                    yq = dy*q

                    xq = xf(yq)


                    xy.append([xq,yq])


                xy = np.array(xy).reshape((len(xy),2))

                ixysort = np.argsort(xy[:,1])

                xy = xy[ixysort,:]


                for i in range(xy.shape[0]-1):

                    x0 = xy[i, 0]
                    x1 = xy[i+1 , 0]

                    y0 = xy[i, 1]
                    y1 = xy[i+1, 1]

                    p = int(np.floor(x0/dx))
                    q = int(np.floor(y0/dy))

                    r = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

                    # B[n + m*N, p+q*Nx] = B[n + m*N, p+q*Nx] + r

                    # BB[0,p+q*Nx] = BB[0,p+q*Nx] + r

                    B[n + m*(N-m), p+q*Nx] = B[n + m*(N-m), p+q*Nx] + r


                xy = []


                for p in range(int(np.ceil(xf(d)/dx)), int(np.floor(xn/dx))):

                    xp = dx*p

                    yp = yb(xp)


                    xy.append([xp, yp])

                for q in range(Ny):

                    yq = dy*q

                    xq = xb(yq)


                    xy.append([xq,yq])


                xy = np.array(xy).reshape((len(xy),2))

                ixysort = np.argsort(xy[:,1])

                ixysort = ixysort[::-1]

                xy = xy[ixysort, :]

                for i in range(xy.shape[0]-1):

                    x0 = xy[i, 0]
                    x1 = xy[i+1 , 0]

                    y0 = xy[i, 1]
                    y1 = xy[i+1, 1]

                    p = int(np.floor(x0/dx))
                    q = int(np.floor(y0/dy))

                    r = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)


                    B[n + m*(N-m), p+q*Nx] = B[n + m*(N-m), p+q*Nx] + r

                    # BB[0,p+q*Nx-1] = BB[0,p+q*Nx-1] + r




        # B = np.array(B).reshape(M,int(Nx*Ny))


        # B = np.array(B).reshape((M,int(Nx*Ny)))

        Gexp = np.array(Gexp).reshape(-1,1)
        Gavg = np.array(Gavg).reshape(-1,1)

        # np.nan_to_num(Gexp,copy=False)
        # np.nan_to_num(Gavg,copy=False)


        Iexp = np.linalg.lstsq(B, Gexp, rcond=rcondnum)[0].reshape((Ny,Nx))

        Iavg = np.linalg.lstsq(B, Gavg, rcond=rcondnum)[0].reshape((Ny,Nx))


        return zoom(Iexp,(int(dy/resolution),ind(dx/resolution)),mode='nearest',order=1), zoom(Iavg,(int(dy/resolution),int(dx/resolution)),mode='nearest',order=1), B
