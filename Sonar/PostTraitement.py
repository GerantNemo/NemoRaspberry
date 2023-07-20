import os
import magic
import numpy as np
import math
import matplotlib.pyplot as plt

from Traitement import Traitement_Sonar
from Clustering import Clustering_Bassin
from Coordonnée import Coordonnée_Mur
from Positionnement import DéplacerLeCentre
from RotationScann import rotationscann, rotationscannQuad

def is_binary_file(file_path):
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)
    return file_type.startswith('application/') or file_type == 'text/plain'

def list_binary_files_in_directory(directory):
    binary_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_binary_file(file_path) and file.lower().endswith('.bin'):
                binary_files.append(file)
    return binary_files

if __name__ == "__main__":
    current_directory = os.getcwd()
    binary_files_list = list_binary_files_in_directory(current_directory)
    Data = []
    Scann =[]
    Map = [np.empty((0, 400))]
    Cpt = 0
    for file_name in binary_files_list:
        print(file_name)
        path = file_name
        file= open(path,"rb")
        data = file.read()

        taille = os.stat(path).st_size
        NombreAngle = int(taille/1024) #Calcule le nombre d'angle dans le fichier avec la longeur de 1024 bits

        for i in range(NombreAngle) :
            #print(data[i*1024:(i+1)*1024])
            Data.append(data[i*1024:(i+1)*1024])

        lim=0.02
        Nb_Point=2
        Distance_point=2/1024

        Traiter=Traitement_Sonar(Data,Distance_point,lim,Nb_Point,NombreAngle)            # Potentiellement un pb avec le nombre d'angle
        Mur,Cluster,Obstacle=Clustering_Bassin(Traiter) 
        Pos,Wall=Coordonnée_Mur(Mur)
        print("J'ai passer le traitement")
        def AngelFound(MurX,MurA,MurB):
            X_IntersectionXA = (MurA[1] - MurX[1])/(MurX[0] - MurA[0])
            X_IntersectionXB = (MurB[1] - MurX[1])/(MurX[0] - MurB[0])

            Y_IntersectionXA = MurX[0] * X_IntersectionXA + MurX[1]
            Y_IntersectionXB = MurX[0] * X_IntersectionXB + MurX[1]

            Intersection=np.array([[X_IntersectionXA , Y_IntersectionXA] , [X_IntersectionXB , Y_IntersectionXB]])

            MinAngel = math.inf

            for i in range(360):
                IntersectionRot=rotationscannQuad(i,Intersection)
                if IntersectionRot[0,0]>0 and IntersectionRot[1,0]>0:
                    Ecart = abs(IntersectionRot[0,0]-IntersectionRot[1,0])
                    if Ecart<MinAngel:
                        MinAngel = Ecart
                        Angel = i
            return Angel

        Angle = AngelFound(Wall[0],Wall[1],Wall[2])
        Orienter = rotationscann(Angle,Traiter)

        Scann=DéplacerLeCentre(Orienter,Pos)

        plt.figure()
        plt.scatter(Scann[:,0],Scann[:,1])
        plt.grid(True)
        plt.title("Scann at position" + str(Pos))

        # ScannRedim = np.empty((0, 400))
        # print(Scann[1])
        # print(ScannRedim[0,0])
        # for i in range(len(Scann)):
        #     ScannRedim[0,i]=Scann[i]

        # Map = np.concatenate((Map,ScannRedim),axis=0)

        Map.append(Scann)
        Cpt = Cpt + 1

    plt.figure("Map Complete")
    for Scann in Map:
        plt.scatter(Scann[:,0],Scann[:,1])
    plt.grid(True)
    plt.title("Complete scann of the sea")
    plt.show()


