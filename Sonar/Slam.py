import numpy as np
import matplotlib.pyplot as plt
import math
import os #Incompatibilité avec linux possible ? tester une fonction plus robuste

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from Traitement import Traitement_Sonar
from RotationScann import rotationscann
from Positionnement import DéplacerLeCentre


def slam(Path1,Path2,Orientation1,Orientation2,Pos1,Pos2):
  """Ce fichier permet à partir de 2 scanns, de leur orientation et de la position à la quel ils onr été pris, de les assemblé 
  dans un seul et unique scann ou tout les positions relative sont correctes.

  Path1=Chemin d'acces vers le premier scann
  Orientation1=dans quel sens était orienter le robot quand il a faire le scann 1
  Pos1=A quel position le robot a fait le scann 1

  Path2=Chemin d'acces vers le deuxième scann
  Orientation2=dans quel sens était orienter le robot quand il a faire le scann 2
  Pos2=A quel position le robot a fait le scann 2    
  """

  #Ouverture des fichier et refermer l'addresse apres
  with open(Path1, 'rb') as file1:
    data = file1.read()
  Un = []
  taille1 = os.stat(Path1).st_size
  NombreAngle1= int(taille1/1024) #Calcule le nombre d'angle dans le fichier avec la longeur de 1024 bits
  for i in range(NombreAngle1) :
    Un.append(data[i*1024:(i+1)*1024])


  with open(Path2, 'rb') as file2:
    data = file2.read()
  Deux= []
  taille2 = os.stat(Path2).st_size
  NombreAngle2= int(taille2/1024) #Calcule le nombre d'angle dans le fichier avec la longeur de 1024 bits
  for i in range(NombreAngle2) :
    #print(data[i*1024:(i+1)*1024])
    Deux.append(data[i*1024:(i+1)*1024])

  #Constante pour traitement du sonar
  lim=0.05
  Nb_Point=3
  Distance_point=2/1024

  #Traitemetn du sonar
  SortieUn=Traitement_Sonar(Un,Distance_point,lim,Nb_Point,NombreAngle1)
  SortieDeux=Traitement_Sonar(Deux,Distance_point,lim,Nb_Point,NombreAngle2)

  #Orienter par rapport au nord nos scann
  OrienterUn=rotationscann(Orientation1,SortieUn)                     #Les angles devrons etre déterminer par rapport à la boussole
  OrienterDeux=rotationscann(Orientation2,SortieDeux)

  #Mettre au bon centre nos scann
  PlacerUn=DéplacerLeCentre(OrienterUn,Pos1)               # Les valeur des centre sont à déterminer soit par le sonar soit pas l'imu
  PlacerDeux=DéplacerLeCentre(OrienterDeux,Pos2)

  #Assembler les scanns
  Map=np.concatenate((PlacerUn,PlacerDeux),axis=1)
  plt.figure()
  plt.scatter(Map[0,:],Map[1,:])

  plt.show()

   

