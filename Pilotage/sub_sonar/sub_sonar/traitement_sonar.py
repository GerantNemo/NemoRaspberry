import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Twist

from brping import Ping360
from datetime import datetime
import numpy as np
from ipywidgets import interact
import math
#from sklearn import cluster, datasets
from scipy.cluster.hierarchy import linkage, fcluster
#from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

class Traitement_sonar(Node):

    def __init__(self):
        super().__init__('traitement_sonar')
        self.publisher_ = self.create_publisher(Twist, 'donnees_sonar', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        #msg = String()
        #msg.data = 'Hello World: %d' % self.i
        #self.publisher_.publish(msg)
        #self.get_logger().info('Publishing: "%s"' % msg.data)
        #self.i += 1

        Distance_point=2/1024
        Scan, NombreAngle = self.ScanSonar(Distance_point)

        lim=0.03
        Nb_Point=5

        Sortie = self.Traitement_Bruit_Sonar(Scan,NombreAngle,Distance_point,lim,Nb_Point,NombreAngle)
        Mur,Cluster,Obstacle = self.Clustering_Bassin(Sortie)
        Angle,Distance = self.Coordonnee(Obstacle)

        twist = Twist()
        twist.linear.x = Distance
        twist.angular.x = Angle
        self.publisher_.publish(twist)

    def ScanSonar(self, Distance_Point):

        interface = "COM3" #'/dev/ttyACM0'

        myPing = Ping360()
        myPing.connect_serial(interface, 115200)


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
    
    def Traitement_Bruit_Sonar(self, sonar_data, Grad, Dist_Unit,limite,Nombre,NombreAngle,DEBUG_AUTOGAIN_OVERRIDE=False,AUTOGAIN_FORCE=False):

        D_max= 250 # Valeur de saturation pour l'autogain

        Res_num= np.empty([1024 ,NombreAngle])
        for i in range(NombreAngle) :
            Res_num[:,i] = np.frombuffer(sonar_data[i], dtype=np.dtype('uint8'))

        #----------------Calcul Auto gain
        argument_max= np.empty([NombreAngle])

        for i in range(NombreAngle): #calcul de la valeur la plus proche pour calcul ou non d'autogain limite
            argument_max[i]=np.argmax(Res_num[i,250:1024])
            #print(Res_num[i,250:1024])

        #print("DEBUG : argument_max",argument_max)
        #print("\n DEBUG : argumentmax max",argument_max.max())

        if(DEBUG_AUTOGAIN_OVERRIDE==False):
            if(argument_max.max()>D_max):
                #print("DEBUG : AUTOGAIN ON")
                limite=limite-0.02
                Nombre=5
            else :
                a = 0 #Ne fait rien
                #print("DEBUG : AUTOGAIN OFF")
        
        elif(AUTOGAIN_FORCE==True):
            #print("DEBUG : AUTOGAIN OVERRIDE FORCE ON")
            limite=limite-0.02
            Nombre=5

        elif(AUTOGAIN_FORCE==False):
            a = 0 #Ne fait rien
            #print("DEBUG : AUTOGAIN OVERRIDE FORCE OFF")

        Res_num=np.transpose(Res_num)

        moyenne_norm= Res_num.mean()/NombreAngle #indicateur normalisé du nombre de points pour autocalcul du gain

        #print(moyenne_norm)

        #limite=moyenne_norm*0.1423317566

        #print("DEBUG : Val_lim =",limite)

        #------------- Fin autogain

        Pres = np.zeros((Grad, 1024))
        Z = np.ones((Res_num.shape[0],)) + 1j*np.ones((Res_num.shape[0],))
        Sortie = np.zeros((Res_num.shape[0],)) 

        Res_num = np.exp(Res_num)
        compteur=0
    
        for sat in range(100, 251, 10):
            for i in range(Res_num.shape[0]):
                for j in range(200, Res_num.shape[1]):
                    if Res_num[i, j] > np.exp(sat):
                        Pres[i, :] = 0
                        Pres[i, j] = 1
                        break

        for i in range(Res_num.shape[0]):
            for k in range(200, Res_num.shape[1]):
                if Pres[i-1, k-1] == 1:
                    Z[i] = -np.cos(i*np.pi*2/Grad)*1j*Dist_Unit*k - np.sin(i*np.pi*2/Grad)*k*Dist_Unit
                    compteur += 1
                    Sortie[i] = k*Dist_Unit
                    break


        ok = np.zeros((compteur,))
        print(compteur)
        for i in range(compteur):
            for j in range(compteur):
                if (np.real(Z[i]) + limite > np.real(Z[j])) and (np.real(Z[j]) > np.real(Z[i]) - limite) and (np.imag(Z[i]) + limite > np.imag(Z[j])) and (np.imag(Z[j]) > np.imag(Z[i]) - limite):
                    ok[i] += 1



        for i in range(compteur):
            if ok[i] < Nombre:
                Z[i] = 0
                Sortie[i] = 0

        SortieBonSens = np.zeros((compteur,))
        for i in range(compteur):
            SortieBonSens[i]=Sortie[compteur-i-1]        

        return SortieBonSens

    def Traitement_Sonar_file(self, file, Grad, Dist_Unit,limite,Nombre):
        Res_num = np.fromfile(file)
        Pres = np.zeros((Grad, 1024))
        Z = np.zeros((Res_num.shape[0],))
        Sortie = np.zeros((Res_num.shape[0],)) 
        Res_num = np.exp(Res_num)
        compteur=0
    
        for sat in range(130, 251, 10):
            for i in range(Res_num.shape[0]):
                for j in range(200, Res_num.shape[1]):
                    if Res_num[i, j] > np.exp(sat):
                        Pres[i, :] = 0
                        Pres[i, j] = 1
                        break

        for i in range(Res_num.shape[0]):
            for k in range(200, Res_num.shape[1]):
                if Pres[i-1, k-1] == 1:
                    Z[i] = -np.cos(i*np.pi*2/Grad)*1j*Dist_Unit*k - np.sin(i*np.pi*2/Grad)*k*Dist_Unit
                    compteur += 1
                    Sortie[i] = k*Dist_Unit
                    break


        ok = np.zeros((compteur,))
        print(compteur)
        for i in range(compteur):
            for j in range(compteur):
                if (np.real(Z[i]) + limite > np.real(Z[j])) and (np.real(Z[j]) > np.real(Z[i]) - limite) and (np.imag(Z[i]) + limite > np.imag(Z[j])) and (np.imag(Z[j]) > np.imag(Z[i]) - limite):
                    ok[i] += 1



        for i in range(compteur):
            if ok[i] < Nombre:
                Z[i] = 0
                Sortie[i] = 0

        SortieBonSens = np.zeros((compteur,))
        for i in range(compteur-1):
            SortieBonSens[i]=Sortie[compteur-i]
        return SortieBonSens
    
    def Clustering_Bassin(self, Sortie):

        X=np.zeros((400,2))
        nb_cluster=8
        Taille_Mur=15
        for i in range(len(Sortie)):
            X[i,0]=math.cos(i*np.pi/200)*Sortie[i]
            X[i,1]=math.sin(i*np.pi/200)*Sortie[i]

        #Marche pas encore si on veux le faire marcher faut remettre les bonnes bibliothèque
        #Affichage_Interactif(X)

        Z = linkage(X,method="single")
        maxdist=max(Z[:,2])  # hauteur du dendrogramme 

        clusters=fcluster(Z, nb_cluster, criterion='maxclust') #Le 5 est le nb de cluster mais faut peut etre le changer en fonction des scans

        #clusters=fcluster(Z, seuil, criterion='distance')              #Si on veux le faire avec un seuil plutot qu'un nb de cluster

        last = Z[-10:, 2]
        last_rev = last[::-1]
        idxs = np.arange(1, len(last) + 1)

        # Récupération des points pour chaque cluster
        num_clusters = np.max(clusters) # Nombre de clusters
        cluster_points = {}
        for i in range(1, num_clusters+1):
            cluster_points[i] = X[clusters == i]

        nombre_points_par_cluster = np.bincount(clusters)
        indice_plus_grand = np.argmax(nombre_points_par_cluster)
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
    
    def Coordonnee(self, MainObstacle):
        Centre=np.mean(MainObstacle,axis=0)
        Distance=norm(Centre)
        Angle=np.arctan2(Centre[1], Centre[0])-np.pi
        return Angle, Distance


def main(args=None):
    rclpy.init(args=args)

    traitement_sonar = Traitement_sonar()

    rclpy.spin(traitement_sonar)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    traitement_sonar.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()