#!/usr/bin/python3
# -*- coding:utf-8 -*-

import csv
import os
import ast

class test_csv():

    def __init__(self):

        #os.chdir("C:\\Users\\T0271829\\Travail\\Projet développement normes\\Donnees_csv")
        #self.path = "C:\\Users\\T0271829\\Travail\\Projet développement normes\\Fichiers CSV"
        self.path = "C:\\Users\\T0271829\\Travail\\Projet développement normes\\Donnees CSV"
        self.namefile = "test.csv"

    def create_class(self):

        classe = (1, 2, 3, 4, 5)
        sous_classe = (("red", "white", "yellow"), (1, 2, 3, 4), (3, 4, 5, 6), ("",), ("",))

        path = ""
        files = ("red.csv", "white.csv", "yellow.csv", "1.csv", "2.csv", "3.csv", "4.csv", "3.csv", "4.csv", "5.csv", "6.csv", "class4.csv", "class5.csv")

        for i in range(len(classe)):
            for j in range(len(sous_classe)):
                chemin_fichier = os.path.join(path, files[i+j])
                self.lire_csv(chemin_fichier, classe[i], sous_classe[i][j])

    def lire_csv(self, chemin, classe, sous_classe):

        etiquettes = []

        with open(chemin, 'r') as f:
            reader = csv.reader(f)

            cpt = 0
            for ligne in reader:
                if cpt > 0:
                    carac_rectangle = ast.literal_eval(ligne[5])
                    name = ligne[0]
                    etiquette = (name, classe, sous_classe, carac_rectangle["x"], carac_rectangle["y"], carac_rectangle["width"], carac_rectangle["height"])
                    etiquettes.append(etiquette)
                cpt += 1

        return etiquettes


if __name__ == '__main__':

    test = test_csv()