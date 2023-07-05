import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

from numpy.linalg import norm
from sklearn.linear_model import RANSACRegressor


def Coordonne_Main_Obstacle(MainObstacle):
    """ à partir du cluster main obstacle trouvée via la fct clustering on renvoie en radian l'angle entre l'avant 
    du sous marin et l'obstacle et la distance les séparants"""
    Centre=np.mean(MainObstacle,axis=0)
    Distance=norm(Centre)
    Angle=np.arctan2(Centre[1], Centre[0])-np.pi
    return Angle, Distance

def Coodonnée_Nautilus(Scann):            #A faire intervenir apres l'orientation et avant le placement
    """ Permet de placer la position du sous marin dans le bassin en utilisant les murs comme reférences
    Scann: Clustering des mur réaliser grace a la fonction cluster"""                    
    #Pour etre sur de l'avoire en carthésien
    ScannCartesien=np.zeros((len(Scann),2))
    if Scann.shape[1]==1:
        for i in range(len(Scann)):
            ScannCartesien[i,0]=Scann[i]*math.cos(i*2*math.pi/400)
            ScannCartesien[i,1]=Scann[i]*math.sin(i*2*math.pi/400)
    else:
        ScannCartesien=Scann
    
    
    # Définir les paramètres de RANSAC
    min_samples = 25  # Nombre minimum de points pour former une ligne
    max_trials = 100  # Nombre d'itérations de RANSAC
    distance_threshold = 0.05  # Distance seuil pour considérer un point comme faisant partie d'une ligne
    plt.figure()
    # Appliquer RANSAC
    # Normalement on peut faire une regression linéaire quartier par quartier pour trouver les lignes puis comparer les equation de droite pour voir les meilleur
    inlinerMax=0
    inlinerSomme=0
    for j in range(int((len(ScannCartesien)/4)-1)):
        print(j)
        print(len(ScannCartesien))
        for i in range(0,3):
            ScannRansac=ScannCartesien[int((((i)*len(ScannCartesien)/4)+j)%len(ScannCartesien)):int((((i+1)*len(ScannCartesien)/4)+j)%len(ScannCartesien)), :]
            ransac = RANSACRegressor(min_samples=min_samples,
                                    residual_threshold=distance_threshold,
                                    max_trials=max_trials)
            ransac.fit(ScannRansac[:, 0].reshape(-1, 1), ScannRansac[:, 1])  # Régression linéaire

            # Obtenir les inliers (points valides) de la meilleure ligne trouvée
            inlier_mask = ransac.inlier_mask_
            inliers = ScannRansac[inlier_mask]
            inlinerSomme=inlinerSomme+len(inliers)
            if i==2:
                if inlinerSomme>inlinerMax:
                    indiceJ=j
                    inlinerMax=inlinerSomme
                inlinerSomme=0
            
                
    for i in range(0,3):
        ScannRansac=ScannCartesien[int((((i)*(len(ScannCartesien))/4)+indiceJ)):int((((i+1)*(len(ScannCartesien))/4)+indiceJ)), :]
        ransac = RANSACRegressor(min_samples=min_samples,
                                    residual_threshold=distance_threshold,
                                    max_trials=max_trials)
        ransac.fit(ScannRansac[:, 0].reshape(-1, 1), ScannRansac[:, 1])  # Régression linéaire

        # Obtenir les inliers (points valides) de la meilleure ligne trouvée
        inlier_mask = ransac.inlier_mask_
        inliers = ScannRansac[inlier_mask]

        plt.scatter(inliers[:, 0], inliers[:, 1], color='r', label='Inliers')
        plt.plot(ScannRansac[:, 0], ransac.predict(ScannRansac[:, 0].reshape(-1, 1)), color='g', label='Ligne détectée')
    # Tracer les lignes détectées
    
    plt.scatter(ScannCartesien[:, 0], ScannCartesien[:, 1], color='b', label='Points')

    plt.legend()
    plt.show()
    return()

    


