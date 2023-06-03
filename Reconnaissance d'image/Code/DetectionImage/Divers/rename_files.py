#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

def rename_images():

    chemin_princ = "D:\\Aymeric\\ESTACA\\Quatrième année\\Associations\\EPIC_ESTACA_SQY\\Pôle Système embarqué\\Concours reconnaissance image\\Images annotées\\rami_marine_dataset"
    classe = "class_"
    cl1_couleur = ("red", "white", "yellow")
    cl2_nombre = ("number_1", "number_2", "number_3", "number_4")
    cl3_nombre = ("number_3", "number_4", "number_5", "number_6")
    cl4 = ("",)
    cl5 = ("",)
    sous_classe = [cl1_couleur, cl2_nombre, cl3_nombre, cl4, cl5]

    for i in range(len(sous_classe)):
        for j in range(len(sous_classe[i])):

            chemin = os.path.join(chemin_princ, classe+str(i+1))
            chemin = os.path.join(chemin, sous_classe[i][j])

            if i == 0:
                nom_sc = sous_classe[i][j] + "_"
            elif i == 1 or i == 2:
                nom_sc = sous_classe[i][j]
                nom_sc = nom_sc.replace('number_','')
                nom_sc += "_"
            else:
                nom_sc = ""

            for k in range(1, len(os.listdir(chemin))+1, 1):

                if k < 10:
                    zero = "000"
                elif 10 <= k < 100:
                    zero = "00"
                elif 100 <= k < 999:
                    zero = "0"
                else:
                    zero = ""

                if i == 4:
                    format = ".jpg"
                else:
                    format = ".png"

                ancien_nom = "img_" + zero + str(k) + format
                nouveau_nom = "img_" + str(i+1) + "_" + str(nom_sc) + str(k) + ".png"

                file_ancien_nom = os.path.join(chemin, ancien_nom)
                file_nouveau_nom = os.path.join(chemin, nouveau_nom)

                print(file_ancien_nom, file_nouveau_nom)

                os.rename(file_ancien_nom, file_nouveau_nom)


if __name__ == "__main__":

    x = 1