import _pickle as _pickle
import FMC
import numpy as np
import os


scan = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoFiles/HighTempExperiments/HighTempDay2/BigHole_RT.p','rb'))

F = FMC.LinearCapture(25,[scan['AScans'][0]],0.6,64,WedgeParameters={'Angle':0,'Height':27,'Velocity':2.978})

A = np.sum(F.AScans[0],axis=1)

F.ProcessScans(50,50)

F.SetRectangularGrid(10,30,80,110,0.1,0.1)

# F.GetWedgeDelays(5.9,0)
#
# delays = F.Delays
#
# print('First Delay Done')
#
# F.GetZeroDegreeWedgeFocusOnReceptionDelays(5.9)
#
# FRdelays = F.Delays
#
# print('Second Delay Done')
#
# TestDelays = {}
# TestDelays['Regular'] = delays
# TestDelays['FocusOnReception'] = FRdelays
# pickle.dump(TestDelays,open('/mnt/c/Users/mmarvasti/Desktop/MoFiles/HighTempExperiments/HighTempDay2/FocusOnReception/Delays103080110.p','wb'))

D = pickle.load(open('/mnt/c/Users/mmarvasti/Desktop/MoFiles/HighTempExperiments/HighTempDay2/FocusOnReception/Delays103080110.p','rb'))

F.Delays = D['Regular']

I = F.ApplyTFM(0)

print('First TFM Done')

del(F)

F = FMC.LinearCapture(25,[A],0.6,64,WedgeParameters={'Angle':0,'Height':27,'Velocity':2.978})

F.ProcessScans2(50,50)

F.SetRectangularGrid(10,30,80,110,0.1,0.1)

F.Delays = (D['FocusOnReception'][0],D['Regular'][0])

II = F.ApplyTFM(0)

print('Second Delay Done')
