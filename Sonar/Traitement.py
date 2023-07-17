import numpy as np
import matplotlib.pyplot as plt

def Traitement_Sonar(sonar_data,Dist_Unit,limite,Nombre,NombreAngle,DEBUG_AUTOGAIN_OVERRIDE=False,AUTOGAIN_FORCE=False):

    """Cette fonction permet d'éliminer les élements qu'il nous semble pas interressant pour notre sous marin:

    sonar_data: Matrice des donnée du scann que l'on aimerai traiter
    Dist_Unit: connaitre la distance entre chaque point renvoyer par le sonar. C'est savoir sur les 1024 points d'une ligne comment 
    chaque point est écarté
    limite: Distance à la quel un point doit avoir des voisins pour etre considérer comme valide
    Nombre: le nombre de voisin que l'on doit prendre en compte dans la distance "limite" pour que le poitn soit valide
    NombreAngle: Nombre de gradiant effectuer par le sonar (en générale 400 pour 1 tour)
    Les valeurs d'apres sont des test pour faire marcher un auto gain et automatiser les valeur de limite mais pour l'instant ca ne marche pas
    """

    #Partis de timoter sur l'autogain mais marche aps pour l'instant

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

    # if(DEBUG_AUTOGAIN_OVERRIDE==False):
    
    #     if(argument_max.max()>D_max):
    #         #print("DEBUG : AUTOGAIN ON")
    #         limite=limite-0.02
    #         Nombre=5
    #     else :
    #         #print("DEBUG : AUTOGAIN OFF")
    # elif(AUTOGAIN_FORCE==True):
    #         #print("DEBUG : AUTOGAIN OVERRIDE FORCE ON")
    #         limite=limite-0.02
    #         Nombre=5
    # elif(AUTOGAIN_FORCE==False):
    #     #print("DEBUG : AUTOGAIN OVERRIDE FORCE OFF")

        

    Res_num=np.transpose(Res_num)


    moyenne_norm= Res_num.mean()/NombreAngle #indicateur normalisé du nombre de points pour autocalcul du gain

    #print(moyenne_norm)

    #limite=moyenne_norm*0.1423317566

    #print("DEBUG : Val_lim =",limite)

    #------------- Fin autogain

    Pres = np.zeros((NombreAngle, 1024))
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
                Z[i] = -np.cos(i*np.pi*2/NombreAngle)*1j*Dist_Unit*k - np.sin(i*np.pi*2/NombreAngle)*k*Dist_Unit
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
    


def Traitement_Sonar_file(file, Grad, Dist_Unit,limite,Nombre):

    """Meme phylosiphie que au dessus masi qui prend en entrée un chemin d'acces du fichhier binaire

    file: Chemin d'acces du fichier de sonar qu'on aimerai traiter
    Grad: Nombre de gradiant effectuer par le sonar (en générale 400 pour 1 tour)
    Dist_Unit: connaitre la distance entre chaque point renvoyer par le sonar. C'est savoir sur les 1024 points d'une ligne comment 
    chaque point est écarté
    limite: Distance à la quel un point doit avoir des voisins pour etre considérer comme valide
    Nombre: le nombre de voisin que l'on doit prendre en compte dans la distance "limite" pour que le poitn soit valide
    Les valeurs d'apres sont des test pour faire marcher un auto gain et automatiser les valeur de limite mais pour l'instant ca ne marche pas
    """

    Res_num = np.fromfile(file)
    #print(Res_num)
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


