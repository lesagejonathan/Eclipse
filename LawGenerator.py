import numpy as np

def CorrectSetting(setting, limits):

    if setting>limits[1]:

        Setting = str(int(limits[1]))

    elif setting<limits[0]:

        Setting = str(int(limits[0]))

    else:

        Setting = str(int(setting))

    return Setting

def ReadBeamToolLawFile(BTFile,ZFile):

    filepath = '/mnt/c/Users/mmarvasti/Desktop/SCCLaws5/'

    f=open(filepath + BTFile,'r')

    L = f.readlines()
    BL = []

    for i in range(len(L)):

        BL.append(L[i].split(' \t'))

    N = BL[1][0]
    A = BL[1][6]
    T = BL[1][8]
    R = BL[1][9]
    S = BL[1][10]
    I = BL[1][11]
    D = BL[1][12]
    V = BL[1][14]

    L = []

    # L.append('V5.0 1\r\n')
    # L.append(N + ' 1000 1 6 1 0 ' + A + ' 900 ' + T + ' ' + R + ' 0 -11999 1015 38302 5920 0\r\n')
    # L.append(N + ' 1000 1 6 1 0 ' + A + ' 2700 ' + T + ' ' + R + ' 0 ' + S + ' ' + I + ' ' + D + ' ' + V + ' 0\n')
    L.append(N + ' 1000 1 6 1 0 0 2700 ' + T + ' ' + R + ' 0 ' + S + ' ' + I + ' ' + D + ' ' + V + ' 0\n')

    elements = list(range(1,int(N)+1))

    for i in range(len(elements)):

        L.append(str(elements[i]) + ' 0 ' + str(BL[elements[i]+1][2]) + ' ' + str(BL[elements[i]+1][3]) + ' 75 100\r\n' )

    L = [LL.encode() for LL in L]

    filepath = '/mnt/c/Users/mmarvasti/Desktop/ZirconLaws/'

    f = open(filepath+ZFile,'wb')
    f.writelines(L)
    f.close()

def CustomLawFile(a):

    L = []

    # L.append('V5.0 ' + str(len(a)) + '\r\n')

    sourcefilepath = '/mnt/c/Users/mmarvasti/Desktop/SCCLaws5/'

    for n in range(len(a)):

        f = open(sourcefilepath + str(a[n]) + '.law','r')

        LL = f.readlines()

        BL = []

        print(n)

        for i in range(len(LL)):

            BL.append(LL[i].split(' \t'))

        N = BL[1][0]
        T = BL[1][8]
        R = BL[1][9]
        V = BL[1][14]

        L.append(N + ' 1000 1 6 1 0 0 2700 ' + T + ' ' + R + ' 0 ' + str(50*n) + ' 0 0 ' + V + ' 0 ' '\n')
        # L.append(N + ' 1000 1 6 1 0 0 2700 ' + T + ' ' + R + ' 0 S ' + I         + ' D ' + V + ' 0\n')

        elements = list(range(1,int(N)+1))

        for i in range(len(elements)):

            L.append(str(elements[i]) + ' 0 ' + str(BL[elements[i]+1][2]) + ' ' + str(BL[elements[i]+1][3]) + ' 75 100\r\n')

        f.close()

    ML = [LL.encode() for LL in L]

    filepath = '/mnt/c/Users/mmarvasti/Desktop/ZirconLaws/CustomLaw.law'

    f = open(filepath,'wb')
    f.writelines(ML)
    f.close()

# def SectorialLawFile(filename, elements, pitch = 0.5, angles=(40, 70), wedgeangle = 39., WedgeVelocity = 2.33, PieceVelocity = 3.24, voltage = 200., gain = 80., pulsewidth = 100.):
#
#     gain = CorrectSetting(gain, (0.,80.))
#     voltage = CorrectSetting(voltage, (50., 200.))
#     pulsewidth = CorrectSetting(pulsewidth, (50., 500.))
#
#     L = []
#
#     angles = range(angles[0], angles[1]+1)
#
#     N = len(elements)
#
#     L.append('V5.0\t'+str(N**2)+'\r\n')
#
#     for i in range(len(angles)):
#
#         header = str(N) + '\t 1000\t1\t'+str(int(40.-20.*np.log10(N)))+'\t1\t0\' + str(angles[i]) \
#         + '0 \t' + str(elements[0]) + '\t' + str(elements[0]) + '\t0\t0\t0\t0\t\' + str(PieceVelocity) + '\r\n'
#
#         L.append(header)
#
#         incidentangle = np.arcsin((WedgeVelocity/PieceVelocity)*np.sin(angles[i]*pi/180)*180/pi
#
#         delayincrement = (pitch/WedgeVelocity)*np.sin(abs(wedgeangle-incidentangle)*pi/180)
#
#         if incidentangle < wedgeangle:
#
#             for j in range(1,N+1):
#
#                 L.append(str(j) + '\t0\' + str(delayincrement*(N-j)) + '\t' + str(delayincrement*(N-j)) + '\t\180\t\50\r\n'  )
#
#         else:
#
#             for j in range(1,N+1):
#
#                 L.append(str(j) + '\t0\' + str(delayincrement*(j-1)) + '\t' + str(delayincrement*(j-1)) + '\t\180\t\50\r\n'  )
#
#     L = [LL.encode() for LL in L]
#
#     f = open(filename,'wb')
#     f.writelines(L)
#     f.close()

def FMCLawFile(filename, elements, voltage = 200., gain = 80., pulsewidth = 100.,ashalfmatrix=True,elmax=None):

    gain = CorrectSetting(gain, (0.,80.))
    voltage = CorrectSetting(voltage, (50., 200.))
    pulsewidth = CorrectSetting(pulsewidth, (50., 500.))

    L = []

    N = len(elements)

    L.append('V5.0\t'+str(N**2)+'\r\n')

    header = '1\t'+'1000\t1\t'+str(int(40.-20.*np.log10(N)))+'\t0\t0\t0\t0\t1\t1\t0\t0\t0\t0\t5900\r\n'


    if (ashalfmatrix)&(elmax is None):

        for m in range(N):

            for n in range(m,N):

                header = '1\t'+'1000\t1\t'+str(int(40.-20.*np.log10(N)))+'\t0\t0\t0\t0\t' + str(elements[m]+1) + '\t' + str(elements[n]+1) + '\t0\t0\t0\t0\t5900\r\n'

                L.append(header)

                L.append('1\t'+gain+'\t0\t0\t'+voltage+'\t'+pulsewidth+'\r\n')

    elif (ashalfmatrix)&(elmax is not None):

        for m in range(N):

            for n in range(m,m+elmax):

                if n<N:

                    header = '1\t'+'1000\t1\t'+str(int(40.-20.*np.log10(N)))+'\t0\t0\t0\t0\t' + str(elements[m]+1) + '\t' + str(elements[n]+1) + '\t0\t0\t0\t0\t5900\r\n'

                    L.append(header)

                    L.append('1\t'+gain+'\t0\t0\t'+voltage+'\t'+pulsewidth+'\r\n')

    else:

        for m in range(N):

            for n in range(N):

                header = '1\t'+'1000\t1\t'+str(int(40.-20.*np.log10(N)))+'\t0\t0\t0\t0\t' + str(elements[m]+1) + '\t' + str(elements[n]+1) + '\t0\t0\t0\t0\t5900\r\n'

                L.append(header)

                L.append('1\t'+gain+'\t0\t0\t'+voltage+'\t'+pulsewidth+'\r\n')

    L = [LL.encode() for LL in L]

    f = open(filename,'wb')
    f.writelines(L)
    f.close()

def FocusOnReception(filename, elements, angle = 0., pitch =0.6, voltage = 200., gain = 80., pulsewidth = 100.):

    gain = CorrectSetting(gain, (0.,80.))
    voltage = CorrectSetting(voltage, (50., 200.))
    pulsewidth = CorrectSetting(pulsewidth, (50., 500.))

    L = []

    N = len(elements)

    L.append('V5.0\t'+str(N)+'\r\n')

    # header = str(N) +'\t'+'1000\t1\t'+str(int(40.-20.*np.log10(N)))+'\t0\t0\t0\t0\t' + str(elements[0]) + '\t' + str(elements[0]) + '\t0\t0\t0\t0\t5900\t0\r\n'

    for i in range(1,N+1):

        header = str(N) +'\t'+'1000\t1\t'+str(int(40.-20.*np.log10(N)))+'\t0\t0\t0\t0\t' + str(elements[0]) + '\t' + str(elements[0]) + '\t0\t' + str(int((i-1)*pitch*1000)) + '\t0\t0\t5900\t0\r\n'

        L.append(header)

        for n in range(1,N+1):

            if (n==i):

                L.append(str(n) + '\t' + gain + '\t0\t0\t' + voltage + '\t' + pulsewidth + '\r\n')

            else:

                L.append(str(n) + '\t' + gain + '\t0\t65535\t' + voltage + '\t' + pulsewidth + '\r\n')


    L = [LL.encode() for LL in L]

    f = open(filename,'wb')
    f.writelines(L)
    f.close()
