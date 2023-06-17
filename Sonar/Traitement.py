import numpy as np
import matplotlib.pyplot as plt

def Traitement_Sonar(sonar_data,Dist_Unit,limite,Nombre,NombreAngle,DEBUG_AUTOGAIN_OVERRIDE=False,AUTOGAIN_FORCE=False):

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
    print("\n DEBUG : argumentmax max",argument_max.max())

    if(DEBUG_AUTOGAIN_OVERRIDE==False):
    
        if(argument_max.max()>D_max):
            print("DEBUG : AUTOGAIN ON")
            limite=limite-0.02
            Nombre=5
        else :
            print("DEBUG : AUTOGAIN OFF")
    elif(AUTOGAIN_FORCE==True):
            print("DEBUG : AUTOGAIN OVERRIDE FORCE ON")
            limite=limite-0.02
            Nombre=5
    elif(AUTOGAIN_FORCE==False):
        print("DEBUG : AUTOGAIN OVERRIDE FORCE OFF")

        

    Res_num=np.transpose(Res_num)


    moyenne_norm= Res_num.mean()/NombreAngle #indicateur normalisé du nombre de points pour autocalcul du gain

    print(moyenne_norm)

    #limite=moyenne_norm*0.1423317566

    print("DEBUG : Val_lim =",limite)

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


