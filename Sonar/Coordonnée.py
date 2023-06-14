import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import norm

testgit=0
def Coordonne(MainObstacle):
    Centre=np.mean(MainObstacle,axis=0)
    Distance=norm(Centre)
    Angle=np.arctan2(Centre[1], Centre[0])-np.pi
    return Angle, Distance