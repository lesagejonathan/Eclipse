import time
import serial


#
# class GalilController:
#
#     import gclib
#
#     def __init__(self,instrument='ImmersionTank'):
#
#
#
#
#         if instrument is 'ImmersionTank':
#
#             self.AxisKeys = {'X':'A','Y':'B','Z':'C','Rotation':'D','E':'Yaw','F':'Pitch'}
#
#             self.StepsPerMeasure = {'X':1000.,'Y':801.721016,'Z':1000.,'Rotation':160.*1000./360.,'Pitch':266.666666667,'Yaw':1310.3}
#
#             ip = '10.10.1.12'
#
#         elif instrument is 'ZMC4':
#
#             self.AxisKeys = {'Rotation':'A', 'Index':'B'}
#
#             self.StepsPerMeasure = {'Index':131345. , 'Rotation':79.6}
#
#             ip = '192.168.1.250'
#
#
#         self.Galil = gclib.py()
#
#         self.Galil.GOpen(ip+' --direct -s ALL')
#
#         self.CurrentPosition = {}
#
#
#         for k in list(self.AxisKeys.keys()):
#
#             self.ZeroEncoder(k)
#
#     def Wait(self):
#
#         inmotion = True
#
#         while inmotion:
#
#             axisinmotion = [self.CheckMotionComplete(axis) for axis in list(self.AxisKeys.keys())]
#
#             inmotion = not(all(axisinmotion))
#
#     def MoveRelative(self,Axis,Position,Speed, Wait=False):
#         # set Speed
#
#         if Wait:
#
#             self.Wait()
#
#
#         self.Galil.GCommand('SH'+self.AxisKeys[Axis])
#
#         self.Galil.GCommand('SP'+self.AxisKeys[Axis]+'='+str(round(Speed*self.StepsPerMeasure[Axis])))
#
#         # set Position
#
#         self.Galil.GCommand('PR'+self.AxisKeys[Axis]+'='+str(round(Position*self.StepsPerMeasure[Axis])))
#
#         # execute Move
#
#         self.Galil.GCommand('BG'+self.AxisKeys[Axis])
#
#
#         # Reset Global Position
#
#         self.CurrentPosition[Axis] += Position
#
#     def MoveAbsolute(self,Axis,Position,Speed,Wait=False):
#
#         if Wait:
#
#             self.Wait()
#
#         # set Speed
#
#         self.Galil.GCommand('SH'+self.AxisKeys[Axis])
#
#         self.Galil.GCommand('SP'+self.AxisKeys[Axis]+'='+str(round(Speed*self.StepsPerMeasure[Axis])))
#
#         # set Position
#
#         self.Galil.GCommand('PA'+self.AxisKeys[Axis]+'='+str(round(Position*self.StepsPerMeasure[Axis])))
#
#         # execute Move
#
#         self.Galil.GCommand('BG'+self.AxisKeys[Axis])
#
#         # Reset Global Position
#
#         self.CurrentPosition[Axis] = Position
#
#     def ZeroEncoder(self,Axis):
#
#         self.Galil.GCommand('DP'+self.AxisKeys[Axis]+'=0')
#         self.CurrentPosition[Axis] = 0.0
#
#     def CheckMotionComplete(self, Axis):
#
#         return not(bool(int(float(self.Galil.GCommand('MG _BG'+self.AxisKeys[Axis])))))
#
#     def MoveToLimit(self, Axis, Speed, Direction, Limit=5000):
#
#
#         if Direction == 'Forward':
#
#                 self.MoveRelative(Axis,Limit,Speed,Wait=True)
#
#                 notlimit = True
#
#                 while notlimit:
#
#                     notlimit = bool(int(float(self.Galil.GCommand('MG _LF'+self.AxisKeys[Axis]))))
#
#
#         elif Direction == 'Backward':
#
#
#                 self.MoveRelative(Axis,-Limit,Speed,Wait=True)
#
#                 notlimit = True
#
#                 while notlimit:
#
#                     notlimit = bool(int(float(self.Galil.GCommand('MG _LR'+self.AxisKeys[Axis]))))
#
#
#     def __del__(self):
#
#         self.Galil.GClose()


class AUTController:

    def __init__(self):

        self.AxisKeys = {'X':'2','Y':'1'}

        self.StepsPerMeasure = {'X':262.307692, 'Y':1424.5}

        self.COM = serial.Serial('/dev/cu.SLAB_USBtoUART')


    def ZeroEncoder(self,axis):

        self.COM.write(('/'+self.AxisKeys[axis]+'ZR\r').encode())

    def SetSpeed(self,axis,speed):

        self.COM.write(('/'+self.AxisKeys[axis]+'V'+str(int(round(self.StepsPerMeasure[axis]*32.768*float(speed))))+'R\r').encode())

    def MoveAbsolute(self,axis,position,speed=None):

        if type(speed) is float:

            self.SetSpeed(axis,speed)

        if position>=0:

            self.COM.write(('/'+self.AxisKeys[axis]+'A'+str(int(round(self.StepsPerMeasure[axis]*float(position))))+'R\r').encode())

        else:

            print('Absolute Motion Must Be Positive')

    def MoveRelative(self,axis,position,speed=None):

        if type(speed) is float:

            self.SetSpeed(axis,speed)

        if position>=0.:

            self.COM.write(('/'+self.AxisKeys[axis]+'P'+str(int(round(self.StepsPerMeasure[axis]*float(position))))+'R\r').encode())

        elif position<0.:

            self.COM.write(('/'+self.AxisKeys[axis]+'D'+str(int(round(self.StepsPerMeasure[axis]*abs(float(position)))))+'R\r').encode())

    # def Jog(self,axis,direction,speed=None):
    #
    #     inpt = None
    #
    #     if direction=='Positive':
    #
    #         position=-1.
    #
    #     elif direction=='Negative':
    #
    #         position=1.
    #
    #     while inpt is None:
    #
    #         self.MoveRelative(axis,position,speed)
    #

    def __del__(self):

        self.COM.close()
