import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import norm
import cv2
from sklearn.linear_model import RANSACRegressor


def Coordonne_Main_Obstacle(MainObstacle):
    Centre=np.mean(MainObstacle,axis=0)
    Distance=norm(Centre)
    Angle=np.arctan2(Centre[1], Centre[0])-np.pi
    return Angle, Distance

def Coodonnée_Nautilus(Scann):            #A faire intervenir apres l'orientation et avant le placement
                        
    #Pour etre sur de l'avoire en carthésien
    ScannCarthésien=np.zeros((len(Scann),2))       
    for i in range(len(Scann)):
        ScannCarthésien[i,0]=Scann[i]*math.cos(i*2*math.pi/400)
        ScannCarthésien[i,1]=Scann[i]*math.sin(i*2*math.pi/400)
    
    
    # Définir les paramètres de RANSAC
    min_samples = 100  # Nombre minimum de points pour former une ligne
    max_trials = 100  # Nombre d'itérations de RANSAC
    distance_threshold = 0.05  # Distance seuil pour considérer un point comme faisant partie d'une ligne

    # Appliquer RANSAC
    ransac = RANSACRegressor(min_samples=min_samples,
                            residual_threshold=distance_threshold,
                            max_trials=max_trials)
    ransac.fit(ScannCarthésien[:, 0].reshape(-1, 1), ScannCarthésien[:, 1])  # Régression linéaire

    # Obtenir les inliers (points valides) de la meilleure ligne trouvée
    inlier_mask = ransac.inlier_mask_
    inliers = ScannCarthésien[inlier_mask]

    # Tracer les lignes détectées
    import matplotlib.pyplot as plt

    plt.scatter(ScannCarthésien[:, 0], ScannCarthésien[:, 1], color='b', label='Points')
    plt.scatter(inliers[:, 0], inliers[:, 1], color='r', label='Inliers')
    plt.plot(ScannCarthésien[:, 0], ransac.predict(ScannCarthésien[:, 0].reshape(-1, 1)), color='g', label='Ligne détectée')
    plt.legend()
    plt.show()
    return()

    


