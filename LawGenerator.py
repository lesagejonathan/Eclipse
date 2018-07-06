import numpy as np

def CorrectSetting(setting, limits):

    if setting>limits[1]:

        Setting = str(int(limits[1]))

    elif setting<limits[0]:

        Setting = str(int(limits[0]))

    else:

        Setting = str(int(setting))

    return Setting


#
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


def FMCLawFile(filename, elements, voltage = 200., gain = 80., pulsewidth = 100.,ashalfmatrix=True):

    gain = CorrectSetting(gain, (0.,80.))
    voltage = CorrectSetting(voltage, (50., 200.))
    pulsewidth = CorrectSetting(pulsewidth, (50., 500.))

    L = []

    N = len(elements)

    L.append('V5.0\t'+str(N**2)+'\r\n')

    header = '1\t'+'1000\t1\t'+str(int(40.-20.*np.log10(N)))+'\t0\t0\t0\t0\t1\t1\t0\t0\t0\t0\t5900\r\n'


    if ashalfmatrix:

        for m in range(N):

            for n in range(m,N):

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
