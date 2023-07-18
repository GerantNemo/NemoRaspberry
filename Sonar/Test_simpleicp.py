
import numpy as np
import matplotlib.pyplot as plt
import math
import os #Incompatibilité avec linux possible ? tester une fonction plus robuste

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from Traitement import Traitement_Sonar
from Clustering import Clustering_Bassin,Clustering_Bassin
from RotationScann import rotationscann
from Positionnement import DéplacerLeCentre
from Coordonnée import Coodonnée_Nautilus,Coordonnée_Mur
from Slam import slam
from simpleicp import PointCloud, SimpleICP

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

Sortie_Bas=rotationscann(-45,Sortie_Bas)            # Quand on tourne le scann de plus de 90° 2 ligne ce superpose

Mur,Cluster,Obstacle=Clustering_Bassin(Sortie_Bas)
Mur_Initial,Cluster_Initial,Obstacle_Intial=Clustering_Bassin(Sortie_Haut)

Pos,LigneMur=Coordonnée_Mur(Mur)                    #LigneMur est une liste de liste pour avoir le premier mur (equation Ax+B) on fait LigneMur[0] et le A=LigneMur[0][0]
PosOrigine,LigneMurOrigine=Coordonnée_Mur(Mur_Initial)

Point=[]



Point = LigneMur[0][0] * Mur[:,0] + LigneMur[0][1] 
Deux = LigneMur[1][0] * Mur[:,0] + LigneMur[1][1] 
Trois = LigneMur[2][0] * Mur[:,0] + LigneMur[2][1] 


XOrigine = LigneMurOrigine[0][0] * Mur_Initial[:,0] + LigneMurOrigine[0][1] 
YOrigine = LigneMurOrigine[1][0] * Mur_Initial[:,0] + LigneMurOrigine[1][1] 
TroisOrigine = LigneMurOrigine[2][0] * Mur_Initial[:,0] + LigneMurOrigine[2][1] 


ScannBas=np.zeros((len(Sortie_Bas),2))
ScannHaut=np.zeros((len(Sortie_Haut),2))

for i in range(len(Sortie_Bas)):
    ScannBas[i,0]=Sortie_Bas[i]*math.cos(i*2*math.pi/400)
    ScannBas[i,1]=Sortie_Bas[i]*math.sin(i*2*math.pi/400)

for i in range(len(Sortie_Haut)):
    ScannHaut[i,0]=Sortie_Haut[i]*math.cos(i*2*math.pi/400)
    ScannHaut[i,1]=Sortie_Haut[i]*math.sin(i*2*math.pi/400)

plt.figure("3 mur bas")
plt.plot(Mur[:,0],Point)
plt.plot(Mur[:,0],Deux)
plt.plot(Mur[:,0],Trois)
plt.scatter(ScannBas[:,0],ScannBas[:,1])
plt.figure("X bas")
plt.plot(Mur[:,0],Point)
plt.scatter(ScannBas[:,0],ScannBas[:,1])

plt.figure("3 mur haut")
plt.plot(Mur_Initial[:,0],XOrigine)
plt.plot(Mur_Initial[:,0],YOrigine)
plt.plot(Mur_Initial[:,0],TroisOrigine)
plt.scatter(ScannHaut[:,0],ScannHaut[:,1])
plt.figure("X haut")
plt.plot(Mur_Initial[:,0],XOrigine)
plt.scatter(ScannHaut[:,0],ScannHaut[:,1])
plt.show()

# nb_lignes, nb_colonnes = Mur.shape
# nb_lignes_Init, nb_colonnes_Init = Mur_Initial.shape

# # Création d'une nouvelle matrice avec une colonne supplémentaire de zéros
# matrice_nouvelle = np.zeros((nb_lignes, 1))
# matrice_nouvelle_Initial = np.zeros((nb_lignes_Init, 1))

# Mur=np.concatenate((Mur,matrice_nouvelle),axis=1)
# Mur_Initial=np.concatenate((Mur_Initial,matrice_nouvelle_Initial),axis=1)

# Origine=PointCloud(Mur_Initial,columns=["x","y","z"])
# Scann=PointCloud(Mur,columns=["x","y","z"])

# icp=SimpleICP()
# icp.add_point_clouds(Origine,Scann)

# H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=1)

# print(H)

