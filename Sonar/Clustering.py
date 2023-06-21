import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def Clustering_Bassin_Polaire(Sortie):

    """Fonction permetant de séparer les mur et les différent obstacle

    Sortie: Scann traiter contenant des donner en géométrie polaire

    Pensée à potentillement adapater les variable:
    nb_cluster: représente le nombre de cluster que l'on cherche dans le scann
    Taille_Mur= le nombre de point d'un cluster pour qu'il soit considéré comme un mur"""

    X=np.zeros((len(Sortie),2))
    nb_cluster=8                    #Variable à adapter au concour
    Taille_Mur=15
    for i in range(len(Sortie)):
        X[i,0]=math.cos(i*np.pi/200)*Sortie[i]
        X[i,1]=math.sin(i*np.pi/200)*Sortie[i]
    

    #Marche pas encpore si on veux le faire marcher faut remettre les bonnes bibliothèque
    #Affichage_Interactif(X)

    Z = linkage(X,method="single")
    maxdist=max(Z[:,2])  # hauteur du dendrogramme 

    
    clusters=fcluster(Z, nb_cluster, criterion='maxclust') #Le 5 est le nb de cluster mais faut peut etre le changer en fonction des scans


    #clusters=fcluster(Z, seuil, criterion='distance')              #Si on veux le faire avec un seuil plutot qu'un nb de cluster
    #plt.figure(figsize=[10,8]);
    #plt.scatter(X[:, 0], X[:, 1], s=40, c=clusters, cmap='jet');
    #plt.title('Single linkage with scipy, seuil='+str(seuil)+', nombres de clusters: '+str(max(clusters)));
    last = Z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)

    

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
    print(indice_plus_grand)
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
    return mur,cluster_points,MainObstacle


def Clustering_Bassin_Carthésien(Sortie):           #Sortie un vecteur avec 2 ligne [X,Y]

    """Fonction permetant de séparer les mur et les différent obstacle

    Sortie: Scann traiter contenant des donner en géométrie carthésienne

    Pensée à potentillement adapater les variable:
    nb_cluster: représente le nombre de cluster que l'on cherche dans le scann
    Taille_Mur= le nombre de point d'un cluster pour qu'il soit considéré comme un mur"""

    X=np.transpose(Sortie)
    nb_cluster=8                    #Variable à adapter au concour
    Taille_Mur=15

    #Marche pas encpore si on veux le faire marcher faut remettre les bonnes bibliothèque
    #Affichage_Interactif(X)

    Z = linkage(X,method="single")
    maxdist=max(Z[:,2])  # hauteur du dendrogramme 

    
    clusters=fcluster(Z, nb_cluster, criterion='maxclust') #Le 5 est le nb de cluster mais faut peut etre le changer en fonction des scans


    #clusters=fcluster(Z, seuil, criterion='distance')              #Si on veux le faire avec un seuil plutot qu'un nb de cluster
    #plt.figure(figsize=[10,8]);
    #plt.scatter(X[:, 0], X[:, 1], s=40, c=clusters, cmap='jet');
    #plt.title('Single linkage with scipy, seuil='+str(seuil)+', nombres de clusters: '+str(max(clusters)));
    last = Z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)

    

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
    print(indice_plus_grand)
    NbpMainObstacle=0
    mur=cluster_points[indice_plus_grand]
    for i in range(1, num_clusters+1):
        if nombre_points_par_cluster[i]>Taille_Mur and i!=indice_plus_grand:
            mur=np.vstack((mur,cluster_points[i]))
        if Taille_Mur>nombre_points_par_cluster[i]>NbpMainObstacle:
            MainObstacle=cluster_points[i]
            NbpMainObstacle=nombre_points_par_cluster[i]
           
            

    plt.figure(figsize=[10,8])
    plt.scatter(X[:, 0], X[:, 1], s=40, c=clusters, cmap='jet')
    plt.title('Single linkage with scipy, n_cluster='+str(nb_cluster))

    x_values = mur[:, 0]
    y_values = mur[:, 1]

    X=MainObstacle[:,0]
    Y=MainObstacle[:,1]
    return mur,cluster_points,MainObstacle
