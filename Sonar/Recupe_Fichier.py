#import imp

from brping import Ping360
import time
from datetime import datetime
from Traitement import Traitement_Sonar
import numpy as np
import matplotlib.pyplot as plt

myPing = Ping360()
myPing.connect_serial("COM3", 115200)


if myPing.initialize() is False:
     print("Failed to initialize Ping!")
     exit(1)


myPing.set_angle(0)


NombreAngle=400
Distance_point=2/1024

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

#print(file.data)

#print("\n")
#print("Array :")
#print(sonar_data)
#data= np.empty([1024 ,NombreAngle])
#for i in range(NombreAngle) :
#     data[:,i] = np.frombuffer(sonar_data[i], dtype=np.dtype('uint8'))
#print(data)
#
#
#
#


lim=0.03
Nb_Point=5
Sortie=Traitement_Sonar(sonar_data,NombreAngle,Distance_point,lim,Nb_Point,NombreAngle)
print(Sortie)
file.close()

theta = np.linspace(0,2*np.pi,NombreAngle)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(theta, Sortie,s=5)
plt.show()
