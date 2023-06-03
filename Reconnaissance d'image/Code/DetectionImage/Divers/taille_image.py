#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2

path = "D:\\Aymeric\\Images_concours"
images_etiquettes = ["images", "etiquettes"]
arborescence = (("red", "white", "yellow"), ("1", "2", "3", "4"), ("3", "4", "5", "6"), ("",), ("",))
etiquettes = ["1_red", "1_white", "1_yellow", "2_1", "2_2", "2_3", "2_4", "3_3", "3_4", "3_5", "3_6", "4", "5"]


def concat_classe_sous_classe(classe, sous_classe):
    if sous_classe != "":
        classe_sous_classe = str(classe) + "_" + str(sous_classe)
    else:
        classe_sous_classe = str(classe)
    return classe_sous_classe


def concat_chemin_image(classe, sous_classe, nom_image=""):

    fichier_classe = "class_" + str(classe)
    if classe == 2 or classe == 3:
        fichier_sous_classe = "number_" + str(sous_classe)
    else:
        fichier_sous_classe = str(sous_classe)

    if fichier_sous_classe != "":
        chemin = os.path.join(path, images_etiquettes[0], fichier_classe, fichier_sous_classe)
    else:
        chemin = os.path.join(path, images_etiquettes[0], fichier_classe)

    if nom_image != "":
        chemin = os.path.join(chemin, nom_image)

    return chemin


def reconstitution_nom_image(numero, classe_sous_classe):
    nom = "img_" + str(classe_sous_classe) + "_" + str(numero) + ".png"
    return nom


def reconstitution_nom_etiquette(classe_sous_classe):
    nom = "label_" + str(classe_sous_classe) + "_" + ".csv"
    return nom

def det_taille_image(numero_image, classe, sous_classe):

    classe_sous_classe = concat_classe_sous_classe(classe, sous_classe)
    nom_image = reconstitution_nom_image(numero_image, classe_sous_classe)
    chemin = concat_chemin_image(classe, sous_classe, nom_image=nom_image)

    img_array = cv2.imread(chemin)  # , cv2.IMREAD_GRAYSCALE)  # convert to array
    taille = (img_array.shape[0], img_array.shape[1])
    return nom_image, taille

def liste_taille_image():

    taille = (720, 1280)

    images = []
    classe = 0

    for cl in arborescence:
        classe += 1
        for sous_classe in cl:
            chemin = concat_chemin_image(classe, sous_classe)
            nbre_image = len(os.listdir(chemin))
            for i in range(1, nbre_image + 1, 1):
                nom_image, taille_image = det_taille_image(i, classe, sous_classe)
                if taille != taille_image:
                    images.append([nom_image, taille_image])

    print(len(images))
    for image in images:
        print(image)

if __name__ == "__main__":

    liste_taille_image()