import numpy as np


def ComputeTRCoeffs(IncidentMode,MaterialProperties):


    cL1 = MaterialProperties['LongitudinalVelocity'][0]

    cL2 = MaterialProperties['LongitudinalVelocity'][1]

    cT1 = MaterialProperties['ShearVelocity'][0]

    cT2 = MaterialProperties['ShearVelocity'][1]

    rho1 = MaterialProperties['Density'][0]

    rho2 = MaterialProperties['Density'][1]

    thi = np.deg2rad(IncidentMode['Angle'])


    mode = IncidentMode['Mode']






    a = np.array([])
