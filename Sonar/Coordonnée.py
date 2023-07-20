import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

from numpy.linalg import norm
from sklearn.linear_model import RANSACRegressor
from Ransac.RANSAC_Algo import fit_walls
from RotationScann import rotationscannQuad


def Coordonne_Main_Obstacle(MainObstacle):
    """ à partir du cluster main obstacle trouvée via la fct clustering on renvoie en radian l'angle entre l'avant 
    du sous marin et l'obstacle et la distance les séparants"""
    Centre=np.mean(MainObstacle,axis=0)
    Distance=norm(Centre)
    Angle=np.arctan2(Centre[1], Centre[0])-np.pi
    return Angle, Distance

def Coodonnée_Nautilus(Scann,Methode=1):            
    #A faire intervenir apres l'orientation et avant le placement
    """ Permet de placer la position du sous marin dans le bassin en utilisant les murs comme reférences
    Scann: Clustering des mur réaliser grace a la fonction cluster.
    La methode 0 cherche les meilleur droite en fonction de leur perpendiculariter
    La méthode 1 cherche les meilleurs droite en fonction du plus grand nombre de point pris en compte
    """                    
    #Pour etre sur de l'avoire en carthésien
    ScannCartesien=np.zeros((len(Scann),2))
    if np.ndim(Scann)==1:
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
    inlinerMax=0
    Line=[[]]
    for j in range(int((len(ScannCartesien)/4)-1)):
        # print(len(ScannCartesien))
        for i in range(0,3):
            Line,inliners=fit_walls(ScannCartesien[int((((i)*len(ScannCartesien)/4)+j)):int((((i+1)*len(ScannCartesien)/4)+j)), :].T)
            if i==0:
                Line0=Line[0]
            if i==1:
                Line1=Line[0]
            if i==2:
                Line2=Line[0]
            inlinerSomme=inlinerSomme+inliners
        
        if Methode==0:
            inlinerSomme=Line0*Line1+Line1*Line2
            distance=abs(inlinerSomme+2)
            if distance<distance_min:
                distance_min=distance
                indiceJ=j  
        if Methode==1:
            if inlinerSomme>inlinerMax:
                indiceJ=j
                inlinerMax=inlinerSomme
            inlinerSomme=0

    # Tracer les lignes détectées          
    for i in range(0,3):
        Line,inliners=fit_walls(ScannCartesien[int((((i)*(len(ScannCartesien))/4)+indiceJ)):int((((i+1)*(len(ScannCartesien))/4)+indiceJ)), :].T)
        PointLine=Line[0]*ScannCartesien[int((((i)*(len(ScannCartesien))/4)+indiceJ)):int((((i+1)*(len(ScannCartesien))/4)+indiceJ)),0]+Line[1]

        plt.plot(ScannCartesien[int((((i)*(len(ScannCartesien))/4)+indiceJ)):int((((i+1)*(len(ScannCartesien))/4)+indiceJ)),0],PointLine)
        if i==0:                                    # Position pas juste
            PosX=math.sin(math.atan(Line[0]))*(-Line[1]/Line[0])    #Egale -0.8
        if i==1:
            PosY=math.sin(math.atan(Line[0]))*(-Line[1]/Line[0])   #Egale -0.7
    print("la position du sous marin en X est:" + str(PosX)+"\n")
    print("la position du sous marin en Y est:" + str(PosY)+"\n")
    plt.scatter(ScannCartesien[:, 0], ScannCartesien[:, 1], color='b', label='Points')
    plt.legend()
    plt.show(block=False)
    return([PosX,PosY])


def Coordonnée_Mur(Scann):            
    #A faire intervenir apres l'orientation et avant le placement
    """ Permet de placer la position du sous marin dans le bassin en utilisant les murs comme reférences
    Scann: Clustering des mur réaliser grace a la fonction cluster.
    La methode 0 cherche les meilleur droite en fonction de leur perpendiculariter
    La méthode 1 cherche les meilleurs droite en fonction du plus grand nombre de point pris en compte
    """                    
    #Pour etre sur de l'avoire en carthésien
    ScannCartesien=np.zeros((len(Scann),2))
    if np.ndim(Scann)==1:
        for i in range(len(Scann)):
            ScannCartesien[i,0]=Scann[i]*math.cos(i*2*math.pi/400)
            ScannCartesien[i,1]=Scann[i]*math.sin(i*2*math.pi/400)
    else:
        ScannCartesien=Scann
    
    # Appliquer RANSAC
    # Normalement on peut faire une regression linéaire quartier par quartier pour trouver les lignes puis comparer les equation de droite pour voir les meilleur
   
    indiceJ=0
    inlinerSomme=0
    LineAll=np.zeros((2,4))                    #Matrice ayant les pente de chaque droite dans la première lignet et l'absicce a l'origine dans la deuxieme ligne. Le nombre de colonne est le nombre de ligne qu'on veux'
    distance_min = float('inf')
    inlinerMax=0
    Line=[]

    L = len(ScannCartesien)

    for j in range(int((L/3)-2)):
        for i in range(0,3):
            Line,inliners=fit_walls(ScannCartesien[int((((i)*L/3)+j)):int((((i+1)*L/3)+j)), :].T)
            inlinerSomme=inlinerSomme+inliners

        if inlinerSomme>inlinerMax:
            indiceJ=j
            inlinerMax=inlinerSomme
        inlinerSomme=0

    # Récupérer et tracer les lignes détectées          
    for i in range(0,3):
        Line,inliners=fit_walls(ScannCartesien[int((((i)*(L)/3)+indiceJ)):int((((i+1)*(L)/3)+indiceJ)), :].T)
        PointLine=Line[0]*ScannCartesien[int((((i)*(L)/3)+indiceJ)):int((((i+1)*(L)/3)+indiceJ)),0]+Line[1]
        if i==0:
            Line0=Line
            Scann0=ScannCartesien[int((((i)*(L)/4)+indiceJ)):int((((i+1)*(L)/4)+indiceJ)), :]
        if i==1:
            Line1=Line
            Scann1=ScannCartesien[int((((i)*(L)/4)+indiceJ)):int((((i+1)*(L)/4)+indiceJ)), :]
        if i==2:
            Line2=Line
            Scann2=ScannCartesien[int((((i)*(L)/4)+indiceJ)):int((((i+1)*(L)/4)+indiceJ)), :]    
        # plt.plot(ScannCartesien[int((((i)*(L)/4)+indiceJ)):int((((i+1)*(L)/4)+indiceJ)),0],PointLine)

    # Pour savoir comment est tournée le repère on cherche le mur étant perpendiculaire aux 2 autres (car pas de mur en face)
    # Pour cela on cherche les 2 droite parallèle via leur coeff directeur

    Perpend0 = abs(Line1[0]-Line2[0])
    Perpend1 = abs(Line0[0]-Line2[0])  
    Perpend2 = abs(Line1[0]-Line0[0])

    Point0 = Line0[0] * Scann[:,0] + Line0[1]
    Point1 = Line1[0] * Scann[:,0] + Line1[1]
    Point2 = Line2[0] * Scann[:,0] + Line2[1]

    plt.figure()
    plt.scatter(Scann0[:,0],Scann0[:,1])
    plt.plot(Scann[:,0],Point0)

    plt.scatter(Scann1[:,0],Scann1[:,1])
    plt.plot(Scann[:,0],Point1)
 
    plt.scatter(Scann2[:,0],Scann2[:,1])
    plt.plot(Scann[:,0],Point2)

    plt.axis('equal')

    

    # MurX sur les autres non
    def MurYSelect(MurX,MurA,MurB):
        X_IntersectionXA = (MurA[1] - MurX[1])/(MurX[0] - MurA[0])
        X_IntersectionXB = (MurB[1] - MurX[1])/(MurX[0] - MurB[0])

        Y_IntersectionXA = MurX[0] * X_IntersectionXA + MurX[1]
        Y_IntersectionXB = MurX[0] * X_IntersectionXB + MurX[1]

        Intersection=np.array([[X_IntersectionXA , Y_IntersectionXA] , [X_IntersectionXB , Y_IntersectionXB]])

        for i in range(360):
            IntersectionRot=rotationscannQuad(i,Intersection)
            if IntersectionRot[0,0]>0 and IntersectionRot[1,0]>0:
                if IntersectionRot[0,1]>IntersectionRot[1,1]:
                    MurY = MurA
                    Mur3 = MurB
                else:
                    MurY = MurB
                    Mur3 = MurA
                return MurY,Mur3

    if Perpend0 < Perpend1 and Perpend0 < Perpend2:
        MurX = Line0
        MurY,Mur3 = MurYSelect(MurX,Line1,Line2)
    elif Perpend1 < Perpend0 and Perpend1 < Perpend2:
        MurX = Line1
        MurY,Mur3 = MurYSelect(MurX,Line2,Line0)
    elif Perpend2 < Perpend0 and Perpend2 < Perpend1:
        MurX = Line2
        MurY,Mur3 = MurYSelect(MurX,Line0,Line1)

    PosY=abs(math.sin(math.atan(MurX[0]))*(-MurX[1]/MurX[0]))   #Egale -0.8
    PosX=abs(math.sin(math.atan(MurY[0]))*(-MurY[1]/MurY[0]))   #Egale -0.7

    # print("la position du sous marin en X est:" + str(PosX)+"\n")
    # print("la position du sous marin en Y est:" + str(PosY)+"\n")
    # plt.scatter(ScannCartesien[:, 0], ScannCartesien[:, 1], color='b', label='Points')
    # plt.legend()
    # plt.show(block=False)
    # print(MurX)
    return([PosX,PosY],[MurX,MurY,Mur3])


    
def OrienterIdeal(MurX,MurA,MurB):
        X_IntersectionXA = (MurA[1] - MurX[1])/(MurX[0] - MurA[0])
        X_IntersectionXB = (MurB[1] - MurX[1])/(MurX[0] - MurB[0])

        Y_IntersectionXA = MurX[0] * X_IntersectionXA + MurX[1]
        Y_IntersectionXB = MurX[0] * X_IntersectionXB + MurX[1]

        Intersection=np.array([[X_IntersectionXA , Y_IntersectionXA] , [X_IntersectionXB , Y_IntersectionXB]])
        for i in range(360):
            IntersectionRot=rotationscannQuad(i,Intersection)
            if IntersectionRot[0,1]/IntersectionRot[0,0]<1 and IntersectionRot[1,1]/IntersectionRot[1,0]<1:
                return i
            
        return 45    
        
            

