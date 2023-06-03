from Traitement import Traitement_Sonar
import numpy as np
import matplotlib.pyplot as plt
import math
import os #Incompatibilité avec linux possible ? tester une fonction plus robuste
from sklearn import cluster, datasets
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors
from Clustering import Clustering_Bassin

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

Sortie_Bas=Traitement_Sonar(Bas,NombreAngle,Distance_point,lim,Nb_Point,NombreAngle)
Sortie_Haut=Traitement_Sonar(Haut,NombreAngle,Distance_point,lim,Nb_Point,NombreAngle)

theta = np.linspace(0,2*np.pi,NombreAngle)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(theta, Sortie_Bas,s=5)
fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
ax2.scatter(theta, Sortie_Haut,s=5)

Mur_Bas,Cluster_Bas,Obstacle_Bas=Clustering_Bassin(Sortie_Bas)
Mur_Haut,Cluster_Haut,Obstacle_Haut=Clustering_Bassin(Sortie_Haut)
plt.show()

