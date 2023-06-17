from Traitement import Traitement_Sonar
import numpy as np
import matplotlib.pyplot as plt
import math
import os #Incompatibilité avec linux possible ? tester une fonction plus robuste
from sklearn import cluster, datasets
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors
from Clustering import Clustering_Bassin_Polaire,Clustering_Bassin_Carthésien
from RotationScann import rotationscann
from Positionnement import DéplacerLeCentre
from Coordonnée import Coodonnée_Nautilus
from Slam import slam

path = "En_Bas.bin"

file= open(path,"rb")


data = file.read()
Bas = []

taille = os.stat(path).st_size
NombreAngle= int(taille/1024) #Calcule le nombre d'angle dans le fichier avec la longeur de 1024 bits

for i in range(NombreAngle) :
    #print(data[i*1024:(i+1)*1024])
    Bas.append(data[i*1024:(i+1)*1024])


path = "En_Haut.bin"

file= open(path,"rb")


data = file.read()
Haut = []

taille = os.stat(path).st_size
NombreAngle= int(taille/1024) #Calcule le nombre d'angle dans le fichier avec la longeur de 1024 bits

for i in range(NombreAngle) :
    #print(data[i*1024:(i+1)*1024])
    Haut.append(data[i*1024:(i+1)*1024])

#Voir tout
lim=0.05
Nb_Point=3
Distance_point=2/1024

Sortie_Bas=Traitement_Sonar(Bas,Distance_point,lim,Nb_Point,NombreAngle)
Sortie_Haut=Traitement_Sonar(Haut,Distance_point,lim,Nb_Point,NombreAngle)

OrienterBas=rotationscann(0,Sortie_Bas)                     #Les angles devrons etre déterminer par rapport à la boussole
OrienterHaut=rotationscann(180,Sortie_Haut)



TaillePiscineX=2                                            # A Adapter au concour peut etre déterminer grace a un premier scann large
TaillePiscineY=1

Coodonnée_Nautilus(OrienterBas)

Mur_Bas,Cluster_Bas,Obstacle_Bas=Clustering_Bassin_Polaire(Sortie_Bas)
Mur_Haut,Cluster_Haut,Obstacle_Haut=Clustering_Bassin_Polaire(Sortie_Haut)


PlacerBas=DéplacerLeCentre(OrienterBas,[0.6,0])               # Les valeur des centre sont à déterminer soit par le sonar soit pas l'imu
PlacerHaut=DéplacerLeCentre(OrienterHaut,[-0.6,0])

Map=np.concatenate((PlacerBas,PlacerHaut),axis=1)
plt.figure()
plt.scatter(Map[0,:],Map[1,:])
Mur,Cluster,Obstacle=Clustering_Bassin_Carthésien(Map)
plt.show()

