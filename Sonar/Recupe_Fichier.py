import time
import numpy as np
import matplotlib.pyplot as plt

from brping import Ping360
from datetime import datetime
from Traitement import Traitement_Sonar

def recupe_fichier():
     """permet de crée un fichier de scann à partir du sonar"""
     myPing = Ping360()
     myPing.connect_serial("COM3", 115200)


     if myPing.initialize() is False:
          print("Failed to initialize Ping!")
          exit(1)


     myPing.set_angle(0)

     NombreAngle=400

     now = datetime.now()
     time_format = "%Y_%m_%d_%H_%M_%S"
     time_str = now.strftime(time_format)+".bin"
     file = open('Sonar_'+time_str,"ab")
     #file = open('teste_Tempsefe',"ab")  #------------------------------------- Changer le nom du fichier ici

     sonar_data = []

     for x in range(NombreAngle) :
          res = myPing.transmitAngle(x)
          file.write(res.data)
          sonar_data.append(res.data)

