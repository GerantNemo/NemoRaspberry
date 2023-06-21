import numpy as np

def rotationscann(Angle,Scann):
    """Permet la rotation du scann par rapport au autre pour se trouvé dans un référentiel absolu
    Angle: De combien de degres on doit tournée notre scann
    Scann: Scann que 'on souhaite tournée"""
    AngleEnRadian=Angle*200/180

    ScannOrientee=np.zeros_like(Scann)

    for i in range(len(Scann)):
        ScannOrientee[i]=Scann[int((i+AngleEnRadian)%len(Scann))]
    return (ScannOrientee)