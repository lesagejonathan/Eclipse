import numpy as np
import FMC
# import misc
import copy



def GetSpectralRange(X,Y,dB=-6):

    return np.where((np.abs(X)>=np.amax(np.abs(X))*10**(dB/20))&((np.abs(Y)>=np.amax(np.abs(Y))*10**(dB/20))))[0]


def FitPowerLaw(f,G,n):

    A = np.hstack(((f**n).reshape(-1,1), np.ones((len(f),1))))

    r = np.linalg.lstsq(A,G.reshape(-1,1))
    #
    # print(A.shape)
    # print(G.shape)

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

        # x = np.linspace(-0.5*p*(N-1),0.5*p*(N-1),N)

        x = np.linspace(0,p*(N-1),N)


        def Integral(m,n,fb):

            if fb == 0:

                a = x[m]
                b = 0.5*(x[n]-x[m])
                c0 = 2*d/(x[n] - 3*x[m])
                c1 = -2*d/(x[n] - 3*x[m])

            elif fb == 1:

                a = 0.5*(x[n]-x[m])
                b = x[n]
                c0 = -2*d/(x[n] + x[m])
                c1 = 2*d/(x[n] + x[m])

            I = np.exp(1j*kx*c1)*(np.exp(1j*b*(kx+ky*c0)) - np.exp(1j*a*(kx+ky*c0)))/(kx+ky*c0)

            infmask = np.isinf(I)

            I[infmask] = np.exp(1j*ky[infmask]*c1)*(b-a)

            return I.reshape(1,-1)

        Gavg = []
        Gexp = []

        B = []

        W = tukey(int(2*windowparams[0]),windowparams[1])

        NFFT = int(windowparams[2]*2*windowparams[0])


        print(NFFT)

        f = np.linspace(0., self.Capture.SamplingFrequency/2., np.floor(NFFT/2) + 1)



        for m in range(N):

            for n in range(m,N):


                ind = int(np.round(self.Capture.SamplingFrequency*np.sqrt((x[n]-x[m])**2 + (2*d)**2)/c))


                NFFT = int(windowparams[2]*2*windowparams[0])

                Aref = rfft(W*aref[m,n,ind-windowparams[0]:ind+windowparams[0]], NFFT)

                A = rfft(W*a[m,n,ind-windowparams[0]:ind+windowparams[0]], NFFT)


                indfreq = GetSpectralRange(Aref,A)

                # print(A.shape)


                v = FitPowerLaw(f[indfreq], np.log(np.abs(A)/np.abs(Aref))[indfreq],fpower)

                Gexp.append(v[0])
                Gavg.append(v[2])

                B.append(Integral(m,n,0) + Integral(m,n,1))




        B = np.array(B).reshape((M,len(kx)))

        print(np.any(np.isnan(B)))
        print(np.any(np.isinf(B)))

        Gavg = np.array(Gavg,dtype=np.complex128).reshape(-1,1)

        Gexp = np.array(Gexp,dtype=np.complex128).reshape(-1,1)

        Binv = np.linalg.pinv(B,rcondnum)

        Aavg = np.dot(Binv,Gavg).reshape(Kx.shape)

        Aexp = np.dot(Binv,Gexp).reshape(Ky.shape)


        Nx = int(np.round((N-1)*p/resolution))

        xzeropad = np.zeros((Aavg.shape[0],int(np.round((Nx - Aavg.shape[1]-2)/2))), dtype=complex)

        Aavg = ifft(ifftshift(np.hstack((xzeropad, Aavg, xzeropad)), axes=(1,)), axis=1)

        Aexp = ifft(ifftshift(np.hstack((xzeropad, Aexp, xzeropad)), axes=(1,)), axis=1)


        Ny = int(np.round(d/(2*resolution)))

        return fftshift(ifft(Aavg,axis=0,n=2*Ny - 2),axes=(1,)), fftshift(ifft(Aexp,axis=0,n=2*Ny - 2),axes=(1,))



    def GetGridImage(self, ScanIndex, RefIndex, d, fpower, c, resolution=0.1, windowparams=(50, 0.1, 4), rcondnum = 1e-2):


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



        M = int(np.round((1+N)*N/2))



        for m in range(self.NumberOfElements):

            xm = m * self.Pitch

            # for n in range(m):

            for n in range(self.NumberOfElements):

            # for n in range(m,self.NumberOfElements):


                xn = n * self.Pitch

                ind = int(round(2 * self.SamplingFrequency *
                                np.sqrt((0.5 * (xn - xm))**2 + d**2) / c))


                # RR = np.max(np.abs(self.AScans[RefIndex][m, n, ind -
                #                                          windowparams[0]:ind -
                #                                          windowparams[0] +
                #                                          int(2 *
                #                                              windowparams[0])]))
                #
                # R = np.max(np.abs(self.AScans[ScanIndex][m, n, ind -
                #                                              windowparams[0]:ind -
                #                                              windowparams[0] +
                #                                              int(2 *
                #                                                  windowparams[0])]))
                #
                #
                indrr = np.argmax(np.abs(aref[m, n, ind -
                                                         windowparams[0]:ind -
                                                         windowparams[0] +
                                                         int(2 *
                                                             windowparams[0])])) + ind - windowparams[0]


                indr = np.argmax(np.abs(a[ScanIndex][m, n, ind -windowparams[0]:ind -windowparams[0] +int(2 *windowparams[0])])) + ind - windowparams[0]


                #

                RR = rfft(w*aref[m, n, indrr -
                                                         windowparams[0]:indrr -
                                                         windowparams[0] +
                                                         int(2 *
                                                             windowparams[0])], int(len(w) *
                                                                                    windowparams[2]))

                R = rfft(w*a[m, n, indr -windowparams[0]:indr - windowparams[0] +int(2 *windowparams[0])], int(len(w) *windowparams[2]))



                RR = RR[indf]
                R = R[indf]
                #
                GG = np.log(np.abs(RR)/np.abs(R))
                # #
                # # GGG.append(GG)
                #
                G[n+m*self.NumberOfElements, 0] = np.trapz(GG/(f**fpower),dx=(f[1]-f[0]))/(fband[1]-fband[0])

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

                    B[n + m * self.NumberOfElements, p+q*Nx] = B[n + m * self.NumberOfElements, p+q*Nx] + r




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

                    B[n + m * self.NumberOfElements, p+q*Nx] = B[n + m * self.NumberOfElements, p+q*Nx] + r




        I = np.linalg.lstsq(B, G, rcond=rcondnum)[0].reshape((Ny,Nx))


        return I
