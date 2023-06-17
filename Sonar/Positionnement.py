import numpy as np
import math
import matplotlib.pyplot as plt
from Clustering import Clustering_Bassin_Polaire,Clustering_Bassin_Carthésien

def PositionnementDuCentre(Scann,TailleX,TailleY):
    Mur,Cluster,Obstacle=Clustering_Bassin_Polaire(Scann)

def DéplacerLeCentre(Scann,PosScann):       # Pos scan est un vecteur vace [PosX,PosY]
    ScanCarthésien=np.zeros((2,len(Scann)))  # Ligne 0 X Ligne1 Y

    for i in range(len(Scann)):
        ScanCarthésien[0,i]=Scann[i]*math.cos(i*2*math.pi/400)
        ScanCarthésien[1,i]=Scann[i]*math.sin(i*2*math.pi/400)
    
    ScanCarthésien[0,:]=ScanCarthésien[0,:]+PosScann[0]
    ScanCarthésien[1,:]=ScanCarthésien[1,:]+PosScann[1]
    
    plt.figure()
    plt.scatter(ScanCarthésien[0,:],ScanCarthésien[1,:])
   

    return(ScanCarthésien)
        