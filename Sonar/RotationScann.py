import numpy as np
import math
import matplotlib.pyplot as plt

def rotationscann(Angle,Scann):
    """Permet la rotation du scann par rapport au autre pour se trouvé dans un référentiel absolu
    Angle: De combien de degres on doit tournée notre scann
    Scann: Scann que 'on souhaite tournée"""
    AngleEnGradian=Angle*200/180

    ScannOrientee=np.zeros_like(Scann)

    for i in range(len(Scann)):
        ScannOrientee[i]=Scann[int((i-AngleEnGradian)%len(Scann))]
    return (ScannOrientee)

def rotationscannQuad(Angle,Scann):
    """
    Permet la rotation d'un scann incomplet en coordonée carthésienne.

    ScannPolaire est un vecteur 2 ligne X colonne
    
    """

    AngleEnRadian=Angle*math.pi/180

    ScannPolaire = np.zeros((len(Scann),2))                                        # [X,0] = angle teta du X eme point,  [X,1] = rayon du X eme point
    ScannOrienter = np.zeros((len(Scann),2))                                        # [X,0] = Coordonnée x du Xeme point, [X,1] = Coordonnée y du Xeme point

    for i in range(len(Scann)):
        ScannPolaire[i,1] = math.sqrt(Scann[i,1]*Scann[i,1]+Scann[i,0]*Scann[i,0])
        if Scann[i,0]>0 and Scann[i,1]>0:                               
            ScannPolaire[i,0] = math.atan(Scann[i,1]/Scann[i,0])                        # Passage en polaire  
        elif Scann[i,0]<0 and Scann[i,1]>0:
            ScannPolaire[i,0] = math.pi-math.atan(Scann[i,1]/abs(Scann[i,0]))
        elif Scann[i,0]<0 and Scann[i,1]<0:
            ScannPolaire[i,0] = math.pi+math.atan(abs(Scann[i,1])/abs(Scann[i,0]))
        elif Scann[i,0]>0 and Scann[i,1]<0:
            ScannPolaire[i,0] = 2*math.pi-math.atan(abs(Scann[i,1])/abs(Scann[i,0]))

        ScannPolaire[i,0] = ScannPolaire[i,0] + AngleEnRadian                       # Décalage des angles

        ScannOrienter[i,0] = math.cos(ScannPolaire[i,0])*ScannPolaire[i,1]
        ScannOrienter[i,1] = math.sin (ScannPolaire[i,0])*ScannPolaire[i,1]

    return ScannOrienter

                    
    