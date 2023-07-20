import numpy as np
import matplotlib.pyplot as plt
import math
import os #Incompatibilité avec linux possible ? tester une fonction plus robuste
import time

from Traitement import Traitement_Sonar
from Coordonnée import Coordonnée_Mur
from Clustering import Clustering_Bassin
from brping import Ping360
from datetime import datetime

def Sonar(Next):

    myPing = Ping360()
    myPing.connect_serial("COM3", 115200)


    if myPing.initialize() is False:
        print("Failed to initialize Ping!")
        exit(1)

    distance =50
    myPing.set_angle(0)
    #myPing.set_transmit_duration(int(distance/1500))              #Regler la distance de scann
    
    NombreAngle=400
    now = datetime.now()
    time_format = "%Y_%m_%d_%H_%M_%S"
    time_str = now.strftime(time_format)+".bin"
    file = open('Sonar_'+time_str,"ab")
  #------------------------------------- Changer le nom du fichier ici
    sonar_data = []

    for x in range(NombreAngle) :
        res = myPing.transmitAngle(x)
        file.write(res.data)
        sonar_data.append(res.data)

    lim=0.05                    # A adapter
    Nb_Point=3
    Distance_point=distance/1024
                                        
    # Traitement du scann
    Traiter=Traitement_Sonar(sonar_data,Distance_point,lim,Nb_Point,NombreAngle)            # Potentiellement un pb avec le nombre d'angle
    Mur,Cluster,Obstacle=Clustering_Bassin(Traiter) 
    Pos,Wall=Coordonnée_Mur(Mur)

    # Déterminer la position de l'obstacle dans le repère Sonar
    if Next == "Center":                        # Retourne la position du centre du repère par rapprot au sous marin (le sous marin sera en 0,0). Sera orienter par rapport à l'avant du sous marin
        CenterX = (Wall[1][1]-Wall[0][1])/(Wall[0][0]-Wall[1][0])
        CenterY = (Wall[0][0]*CenterX+Wall[0][1])

        PosNext = [[CenterX],[CenterY]]
    elif Next == "Main_Obstacle":
        TotX = 0
        TotY = 0

        for i in range(len(Obstacle)):
            TotX = TotX + Obstacle[i][0]
            TotY = TotY + Obstacle[i][1]

        PosNextX = TotX / len(Obstacle)
        PosNextY = TotY / len(Obstacle)

        PosNext = [[PosNextX],[PosNextY]]
    elif Next == "Boue":
        CentreCluster=[]
        TotX = []
        TotY = []
        PosX = []
        PosY = []
        NormMin = math.inf

        for i in range(len(Cluster)):
            for j in range(len(Cluster[i])):
                TotX[i] = TotX[i] + Cluster[i][j][0]
                TotY[i] = TotY[i] + Cluster[i][j][0]
            PosX[i] = TotX[i]/len(Cluster[i])
            PosY[i] = TotY[i]/len(Cluster[i])
            CentreCluster[i] = [[PosX[i]],[PosY[i]]]
            Norm = math.sqrt(PosX[i]*PosX[i]+PosY[i]*PosY[i])
            if Norm < NormMin:
                NormMin = Norm
                PosNext = CentreCluster[i]
    else:
        PosNext=[[0],[0]]
    
    # Détermine l'angle de rotation
    Angle = 0
    if PosNext[0]<0:
        Angle = math.atan(PosNext[1]/-PosNext[0])
    elif PosNext[0]>0 and PosNext[1]>0:
        Angle = math.atan(PosNext[1]/PosNext[0])+(math.pi/2)
    elif PosNext[0]>0 and PosNext[1]<0:
        Angle = math.atan(PosNext[1]/PosNext[0])-(math.pi/2)
    elif PosNext[0]==0 and PosNext[1]>0:
        Angle = math.pi/2
    elif PosNext[0]==0 and PosNext[1]<0:
        Angle = -math.pi/2
    elif PosNext[1]==0 and PosNext[0]>0:
        Angle = math.pi
    elif PosNext[1]==0 and PosNext[0]<0:
        Angle = 0

    # Transforme la position dans le repère Bassin
    PosNext[0] = abs(math.sin(math.atan(Wall[1][0]))*(-Wall[1][1]/Wall[1][0])) + PosNext[0]
    PosNext[1] = abs(math.sin(math.atan(Wall[0][0]))*(-Wall[0][1]/Wall[0][0])) + PosNext[1]

    return Pos,PosNext,Angle








