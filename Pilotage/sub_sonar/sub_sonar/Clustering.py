
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import cluster, datasets
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors


# pour les "widgets" Jupyter permettant de régler les valeurs de variables 
#import nbimporter
#from AffichargeInteractif import Affichage_Interactif
import ipywidgets as widgets  
from ipywidgets import interact

def Clustering_Bassin(Sortie):
    X=np.zeros((400,2))
    nb_cluster=8
    Taille_Mur=15
    for i in range(len(Sortie)):
        X[i,0]=math.cos(i*np.pi/200)*Sortie[i]
        X[i,1]=math.sin(i*np.pi/200)*Sortie[i]
    

    #Marche pas encpore si on veux le faire marcher faut remettre les bonnes bibliothèque
    #Affichage_Interactif(X)

    Z = linkage(X,method="single")
    maxdist=max(Z[:,2])  # hauteur du dendrogramme 
    #plt.figure(figsize=[10,8])
    #dendrogram(Z) #,truncate_mode="level",p=10);  # le paramètre p permet éventuellement de ne pas afficher le "bas" du dendrogramme, utile pour un grand jeu de données
    #plt.title('Single linkage dendrogram with scipy')
    
    clusters=fcluster(Z, nb_cluster, criterion='maxclust') #Le 5 est le nb de cluster mais faut peut etre le changer en fonction des scans
    #plt.figure(figsize=[10,8])
    #plt.scatter(X[:, 0], X[:, 1], s=40, c=clusters, cmap='jet')
    #plt.title('Single linkage with scipy, n_cluster='+str(nb_cluster))

    #clusters=fcluster(Z, seuil, criterion='distance')              #Si on veux le faire avec un seuil plutot qu'un nb de cluster
    #plt.figure(figsize=[10,8]);
    #plt.scatter(X[:, 0], X[:, 1], s=40, c=clusters, cmap='jet');
    #plt.title('Single linkage with scipy, seuil='+str(seuil)+', nombres de clusters: '+str(max(clusters)));
    last = Z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    #plt.figure(figsize=(10, 5))
    #plt.title('Méthode du coude')
    #plt.xlabel('Nombre de clusters')
    #plt.ylabel('Distance')
    #plt.plot(idxs, last_rev, 'k-', lw=2)
    #plt.xticks(idxs)
    

    # Récupération des points pour chaque cluster
    num_clusters = np.max(clusters) # Nombre de clusters
    cluster_points = {}
    for i in range(1, num_clusters+1):
        cluster_points[i] = X[clusters == i]

 

    # Affichage des points pour chaque cluster
    #for i in range(1, num_clusters+1):
        #print('Cluster', i, ' : ', cluster_points[i])
    
    
    nombre_points_par_cluster = np.bincount(clusters)
    indice_plus_grand = np.argmax(nombre_points_par_cluster)
    #print(indice_plus_grand)
    NbpMainObstacle=0
    mur=cluster_points[indice_plus_grand]
    for i in range(1, num_clusters+1):
        if nombre_points_par_cluster[i]>Taille_Mur and i!=indice_plus_grand:
            mur=np.vstack((mur,cluster_points[i]))
        if Taille_Mur>nombre_points_par_cluster[i]>NbpMainObstacle:
            MainObstacle=cluster_points[i]
            NbpMainObstacle=nombre_points_par_cluster[i]
           
            


    x_values = mur[:, 0]
    y_values = mur[:, 1]

    X=MainObstacle[:,0]
    Y=MainObstacle[:,1]
    #plt.figure(figsize=[10,8])
    #plt.scatter(x_values,y_values)
    #plt.scatter(X,Y)
    #plt.title('Mur du bassin')

    return mur,cluster_points,MainObstacle