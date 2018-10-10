import numpy as np


#
# def EffectiveAperture(X,Y,x):
#
#     x = np.array(x)
#
#     L = np.sqrt((X-x)**2 + Y**2)
#
#     dx = x[-1] - x[0]
#
#     alpha = np.arccos((L[0]**2 + L[1]**2 - dx**2)/(2*L[0]*L[1]))
#
#     # alpha = np.arcsin(Y*dx/(np.amin(L)*np.amax(L)))
#
#     # print(np.rad2deg(alpha))
#     #
#     # a = np.sqrt(2*np.amin(L)**2 - 2*np.amin(L)*np.cos(alpha))
#
#     a = np.amin(L)*np.sin(alpha/2)*2
#
#     z = np.amin(L)*np.cos(alpha/2)
#
#
#     return a,z






def NumericalAperture(x,y,L):

    if (type(x) is not float)&(type(y) is not float):

        x,y = np.meshgrid(x,y)

    rp = np.sqrt((x - L/2)**2 + y**2)

    rm = np.sqrt((x + L/2)**2 + y**2)

    A = np.sin(0.5*np.arccos((rp**2 + rm**2 - L**2)/(2*rp*rm)))

    A[0,:] = A[1,:]

    return A


def FocusedPressure(D,r,z,c,f):

    lmbd = c/f

    h = r - np.sqrt(r**2 - 0.25*D**2)

    p = abs(2/(1-z/r))*abs(np.sin((np.pi/lmbd)*(np.sqrt((z-h)**2 + 0.25*D**2))))

    return p



def NearFieldLength(a,b,c,f):

    """ Computes approximate nearfield length for rectangular probes with size axb; a>b """

    hi = np.array([0.99,0.99,0.99,1.,1.01,1.04,1.09,1.15,1.25,1.37])

    ba = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

    h = np.interp(b/a,ba,hi,hi[0],hi[-1])

    return h*((0.5*a)**2/(c/f))

def FocusFactor(a,b,c,f,zf):

    K = zf/NearFieldLength(a,b,c,f)

    if K<1.:

        return K

    else:

        return 1.

def BeamDiameter(a,b,c,f,zf,fieldtype='Echo'):

    """ Computes the approximate beam diameter (-6dB) at specied focal distance, zf """

    N = NearFieldLength(a,b,c,f)

    K = FocusFactor(a,b,c,f,zf)

    k ={'Echo':0.51,'Free':0.7}

    lmbd = c/f

    k = k[fieldtype]

    b = (k*lmbd/a)**2

    BD = K*np.sqrt((-4*N**2*b)/(b - 1.))
    #
    # BD = K*k[fieldtype]*zf/(2*N)

    # BD = K*k*zf/(2*N)

    return BD

def ContactFMCQuality(b,f,N,p,c,xgrid,ygrid):

    Kgrid = np.zeros((len(ygrid),len(xgrid)))

    BDgrid = np.zeros((len(ygrid),len(xgrid)))

    NAgrid = np.zeros((len(ygrid),len(xgrid)))


    for ix in range(len(xgrid)):

        for iy in range(len(ygrid)):

            a,zf = EffectiveAperture(xgrid[ix],ygrid[iy],[-p*0.5*(N-1), p*0.5*(N-1)])

            # NF = NearFieldLength(a,b,c,f)

            BDgrid[iy,ix] = BeamDiameter(a,b,c,f,zf)

            # Agrid[iy,ix] = (1/FocusFactor(a,b,c,f,zf))**2

            Agrid[iy,ix] = FocusFactor(a,b,c,f,zf)


    return Agrid,BDgrid
