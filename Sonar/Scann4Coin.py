import numpy as np
import matplotlib.pyplot as plt
import math
import os #Incompatibilité avec linux possible ? tester une fonction plus robuste
import time

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from Traitement import Traitement_Sonar
from RotationScann import rotationscann
from Positionnement import DéplacerLeCentre
from Coordonnée import Coordonnée_Mur
from Clustering import Clustering_Bassin
from brping import Ping360
from datetime import datetime
from simpleicp import PointCloud, SimpleICP

def Sonar4coin():

    myPing = Ping360()
    myPing.connect_serial("COM12", 115200)


    if myPing.initialize() is False:
        print("Failed to initialize Ping!")
        exit(1)

    distance =50
    myPing.set_angle(0)
    myPing.set_transmit_duration(int(distance/1500))              #Regler la distance de scann
    

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

    Traiter=Traitement_Sonar(sonar_data,Distance_point,lim,Nb_Point,NombreAngle)

    Mur,Cluster,Obstacle=Clustering_Bassin(Traiter)

    PosInit,Wall=Coordonnée_Mur(Mur)

    CenterX=(Wall[1][1]-Wall[0][1])/(Wall[0][0]-Wall[1][0])
    CenterY=(Wall[0][0]*CenterX+Wall[0][1])

    PosCenter=[[CenterX],[CenterY]]

    return(PosInit,PosCenter)

