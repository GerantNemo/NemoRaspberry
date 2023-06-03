from brping import Ping360
import time
from datetime import datetime
import numpy as np
import os

def ScannSonar(Distance_Point):
    myPing = Ping360()
    myPing.connect_serial("COM3", 115200)


    if myPing.initialize() is False:
        print("Failed to initialize Ping!")
        exit(1)


    myPing.set_angle(0)
    #myPing.set_transmit_duration(int(25/1500))
    NombreAngle=400
    now = datetime.now()
    time_format = "%Y_%m_%d_%H_%M_%S"
    time_str = now.strftime(time_format)+".bin"
    file = open('Sonar_'+time_str,"ab")
  #------------------------------------- Changer le nom du fichier ici
    sonar_data = []

    for x in range(NombreAngle) :
        res = myPing.transmitAngle(x)
        file.write(res.data)
        sonar_data.append(res.data)

    return sonar_data,NombreAngle

