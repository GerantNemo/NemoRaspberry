import numpy as np

def rotationscann(Angle,Scann):
    AngleEnRadian=Angle*200/180

    ScannOrientee=np.zeros_like(Scann)

    for i in range(len(Scann)):
        ScannOrientee[i]=Scann[int((i+AngleEnRadian)%len(Scann))]
    return (ScannOrientee)