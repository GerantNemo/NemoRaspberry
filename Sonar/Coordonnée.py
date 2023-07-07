import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

from numpy.linalg import norm
from sklearn.linear_model import RANSACRegressor
from Ransac.RANSAC_Algo import fit_walls


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
    
    plt.figure()
    
    # Appliquer RANSAC
    # Normalement on peut faire une regression linéaire quartier par quartier pour trouver les lignes puis comparer les equation de droite pour voir les meilleur
   
    indiceJ=0
    inlinerSomme=0
    LineAll=np.zeros((2,4))                    #Matrice ayant les pente de chaque droite dans la première lignet et l'absicce a l'origine dans la deuxieme ligne. Le nombre de colonne est le nombre de ligne qu'on veux'
    distance_min = float('inf')
    for j in range(int((len(ScannCartesien)/4)-1)):
        # print(len(ScannCartesien))
        for i in range(0,3):
            Line,inliners=fit_walls(ScannCartesien[int((((i)*len(ScannCartesien)/4)+j)):int((((i+1)*len(ScannCartesien)/4)+j)), :].T)
            if i==0:
                Line0=Line[0]
            if i==1:
                Line1=Line[0]
            #inlinerSomme=inlinerSomme+inliers
            if i==2:
                inlinerSomme=Line0*Line1+Line1*Line[0]
                distance=abs(inlinerSomme+3)
                if distance<distance_min:
                    distance_min=distance
                    indiceJ=j


            # if inlinerSomme>inlinerMax[i]:
            #     inlinerMax[i]=inliners
            #     indiceJ[i]=j

            # if i==2:
            #     if inlinerSomme>inlinerMax:
            #         indiceJ=j
            #         inlinerMax=inlinerSomme
            #     inlinerSomme=0
        
            
                
    for i in range(0,3):
        Line,inliners=fit_walls(ScannCartesien[int((((i)*(len(ScannCartesien))/4)+indiceJ)):int((((i+1)*(len(ScannCartesien))/4)+indiceJ)), :].T)
        PointLine=Line[0]*ScannCartesien[int((((i)*(len(ScannCartesien))/4)+indiceJ)):int((((i+1)*(len(ScannCartesien))/4)+indiceJ)),0]+Line[1]

        plt.plot(ScannCartesien[int((((i)*(len(ScannCartesien))/4)+indiceJ)):int((((i+1)*(len(ScannCartesien))/4)+indiceJ)),0],PointLine)
    # # Tracer les lignes détectées
    
    plt.scatter(ScannCartesien[:, 0], ScannCartesien[:, 1], color='b', label='Points')

    plt.legend()
    plt.show()
    return()

    


