import numpy as np
import math
import matplotlib.pyplot as plt


def DéplacerLeCentre(Scann,PosScann):       # Pos scan est un vecteur vace [PosX,PosY]
    """ permet de déplacer le centre du scann a sont emplacement réel par rapport au autre scann effectuées.
    Scann: Est le scann que l'on souhaite déplacer
    PosScann: positon [X,Y] de ou a été pris le scann dans la piscine"""
    ScanCarthesien=np.zeros((len(Scann),2))  # Colonne 0 X Colonne Y

    if np.ndim(Scann)==1:
        for i in range(len(Scann)):
            ScanCarthesien[i,0]=Scann[i]*math.cos(i*2*math.pi/400)
            ScanCarthesien[i,1]=Scann[i]*math.sin(i*2*math.pi/400)
    else:
        ScanCarthesien=Scann
    
    ScanCarthesien[:,0]=ScanCarthesien[:,0]+PosScann[0]
    ScanCarthesien[:,1]=ScanCarthesien[:,1]+PosScann[1]
    
    plt.figure()
    plt.scatter(ScanCarthesien[:,0],ScanCarthesien[:,1])
   

    return(ScanCarthesien)
        