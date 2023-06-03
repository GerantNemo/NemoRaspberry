from Traitement import Traitement_Sonar
import numpy as np
import matplotlib.pyplot as plt
import math
import os #Incompatibilité avec linux possible ? tester une fonction plus robuste
from sklearn import cluster, datasets
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors
from Clustering import Clustering_Bassin
from Coordonnée import Coordonne


# pour les "widgets" Jupyter permettant de régler les valeurs de variables 
#import nbimporter
#from AffichargeInteractif import Affichage_Interactif
import ipywidgets as widgets  
from ipywidgets import interact

#path= "../Fichier sonar/Sonar_2023_02_16_14_59_14.bin"

path = "En_Bas.bin"

file= open(path,"rb")


data = file.read()
res = []

taille = os.stat(path).st_size
NombreAngle= int(taille/1024) #Calcule le nombre d'angle dans le fichier avec la longeur de 1024 bits

for i in range(NombreAngle) :
    #print(data[i*1024:(i+1)*1024])
    res.append(data[i*1024:(i+1)*1024])

#print(res)

#Voir tout
lim=0.05
Nb_Point=3

#Voir que les murs
limMur=0.03
Nb_Point_Mur=8
Distance_point=2/1024


Sortie=Traitement_Sonar(res,NombreAngle,Distance_point,lim,Nb_Point,NombreAngle)

theta = np.linspace(0,2*np.pi,NombreAngle)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(theta, Sortie,s=5)

Mur,Cluster,Obstacle=Clustering_Bassin(Sortie)

Angle,Disance=Coordonne(Obstacle)

print(Angle)
print(Disance)

