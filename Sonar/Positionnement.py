import numpy as np
import math
import matplotlib.pyplot as plt

from Clustering import Clustering_Bassin_Polaire,Clustering_Bassin_Carthésien

def PositionnementDuCentre(Scann,TailleX,TailleY):
    Mur,Cluster,Obstacle=Clustering_Bassin_Polaire(Scann)

def DéplacerLeCentre(Scann,PosScann):       # Pos scan est un vecteur vace [PosX,PosY]
    """ permet de déplacer le centre du scann a sont emplacement réel par rapport au autre scann effectuées.
    Scann: Est le scann que l'on souhaite déplacer
    PosScann: positon [X,Y] de ou a été pris le scann dans la piscine"""
    ScanCarthesien=np.zeros((2,len(Scann)))  # Ligne 0 X Ligne1 Y

    for i in range(len(Scann)):
        ScanCarthesien[0,i]=Scann[i]*math.cos(i*2*math.pi/400)
        ScanCarthesien[1,i]=Scann[i]*math.sin(i*2*math.pi/400)
    
    ScanCarthesien[0,:]=ScanCarthesien[0,:]+PosScann[0]
    ScanCarthesien[1,:]=ScanCarthesien[1,:]+PosScann[1]
    
    plt.figure()
    plt.scatter(ScanCarthesien[0,:],ScanCarthesien[1,:])
   

    return(ScanCarthesien)
        