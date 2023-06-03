#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
#import tensorflow as tf
import ast
#import matplotlib.pyplot as plt
import cv2
import csv
#import pandas as p

def open_br_eti_csv():

    path = "D:\\Aymeric\\Banque_images_reduites\\etiquettes"
    chemin_couleur = ["label_red.csv", "label_white.csv", "label_yellow.csv"]
    classe = "1"
    sous_classe = ["red", "white", "yellow"]
    labels = []

    for i in range(len(chemin_couleur)):
        chemin = os.path.join(path, chemin_couleur[i])
        labels.append(lire_csv(chemin, classe, sous_classe[i]))
    return labels

def lire_csv_br(chemin, classe, sous_classe):

    etiquettes = []

    with open(chemin, newline='', encoding ='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

        cpt = 0
        for ligne in reader:
            if cpt > 0:
                carac_circle = ast.literal_eval(ligne[5])
                name = ligne[0]
                #etiquette = (name, classe, sous_classe, carac_rectangle["x"], carac_rectangle["y"], carac_rectangle["width"], carac_rectangle["height"])
                etiquette = (name, classe, sous_classe, carac_circle["cx"], carac_circle["cy"])
                etiquettes.append(etiquette)
            cpt += 1

def lire_csv(chemin, classe, sous_classe):

    etiquettes = []

    if classe != 5:

        with open(chemin, newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

            cpt = 0
            for ligne in reader:
                if cpt > 0:
                    carac = ast.literal_eval(ligne[5])
                    name = ligne[0]
                    if classe == 1:
                        etiquette = (name, classe, sous_classe, carac["cx"], carac["cy"])
                    else:
                        cx = carac["x"] + carac["width"] / 2
                        cy = carac["y"] + carac["height"] / 2
                        etiquette = (name, classe, sous_classe, cx, cy)
                        # etiquette = (name, classe, sous_classe, carac_rectangle["x"], carac_rectangle["y"], carac_rectangle["width"], carac_rectangle["height"])

                    etiquettes.append(etiquette)
                cpt += 1

    return etiquettes

if __name__ == "__main__":

    labels = open_br_eti_csv()
    print(labels)