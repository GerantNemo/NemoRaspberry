from Traitement import Traitement_Sonar
import numpy as np
import math
from Clustering import Clustering_Bassin
from Coordonnée import Coordonne
from Fichier_Array import ScannSonar


Distance_point=2/1024

Scann, NombreAngle=ScannSonar(Distance_point)

lim=0.03
Nb_Point=5

Sortie=Traitement_Sonar(Scann,NombreAngle,Distance_point,lim,Nb_Point,NombreAngle)
Mur,Cluster,Obstacle=Clustering_Bassin(Sortie)
Angle,Disance=Coordonne(Obstacle)
print(Angle)
print(Disance)