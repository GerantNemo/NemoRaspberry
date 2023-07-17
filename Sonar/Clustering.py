import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def Clustering_Bassin(Sortie,Mur_Grand=0):           

    """Fonction permetant de séparer les mur et les différent obstacle

    Sortie: Scann traiter contenant des donner en géométrie carthésienne

    Pensée à potentillement adapater les variable:
    nb_cluster: représente le nombre de cluster que l'on cherche dans le scann
    Taille_Mur= le nombre de point d'un cluster pour qu'il soit considéré comme un mur
    
    Return:
    Mur: La position en carthésien point par point des mur (un point sur une ligne chaque ligne un point différents) """

    
    nb_cluster=8                    #Variable à adapter au concour
    Taille_Mur=Mur_Grand
    if np.ndim(Sortie)==1:
        X=np.zeros((len(Sortie),2))
        for i in range(len(Sortie)):
            X[i,0]=math.cos(i*np.pi/200)*Sortie[i]
            X[i,1]=math.sin(i*np.pi/200)*Sortie[i]
    else:
        X=Sortie

    Z = linkage(X,method="single")
    
    clusters=fcluster(Z, nb_cluster, criterion='maxclust') 

    # Récupération des points pour chaque cluster
    num_clusters = np.max(clusters) # Nombre de clusters
    cluster_points = {}
    for i in range(1, num_clusters+1):
        cluster_points[i] = X[clusters == i]
    
    nombre_points_par_cluster = np.bincount(clusters)
    indice_plus_grand = np.argmax(nombre_points_par_cluster)

    NbpMainObstacle=0
    mur=cluster_points[indice_plus_grand]
    if Mur_Grand!=0:
        for i in range(1, num_clusters+1):
            if nombre_points_par_cluster[i]>Taille_Mur and i!=indice_plus_grand:
                mur=np.vstack((mur,cluster_points[i]))
    else:
        Taille_Mur=np.argmax(np.bincount(clusters))
    for i in range(1, num_clusters+1):   
        if Taille_Mur>nombre_points_par_cluster[i]>NbpMainObstacle:
                MainObstacle=cluster_points[i]
                NbpMainObstacle=nombre_points_par_cluster[i]
           
            

    # plt.figure(figsize=[10,8])
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=clusters, cmap='jet')
    # plt.title('Single linkage with scipy, n_cluster='+str(nb_cluster))
    # plt.show()
    return mur,cluster_points,MainObstacle
