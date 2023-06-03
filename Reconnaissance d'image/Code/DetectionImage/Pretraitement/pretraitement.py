#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import csv
import pandas as p
import numpy as np
import scipy

from DetectionImage.Divers import open_file

class DataSetCreator(object):

    def __init__(self, batch_size, image_height, image_width):#, dataset):
        self.classname = ["classe", "sous_classe", "centre_x", "centre_y"]
        self.batch_size = batch_size
        self.image_original_height = 720
        self.image_original_width = 1280
        self.image_height = image_height
        self.image_width = image_width
        #self.dataset = tf.data.Dataset()

        self.path = "D:\\Aymeric\\Images_concours"
        self.images_etiquettes = ["images", "etiquettes"]
        self.arborescence = (("red", "white", "yellow"), )#, ("1", "2", "3", "4"), ("3", "4", "5", "6"), ("",), ("",))
        self.etiquettes = ["1_red", "1_white", "1_yellow"]#, "2_1", "2_2", "2_3", "2_4", "3_3", "3_4", "3_5", "3_6", "4", "5"]

        self.chemin_images = pathlib.WindowsPath("D:\\Aymeric\\Banque_images_reduites\\images")
        self.classes = np.array([item.name for item in self.chemin_images.glob('*')])
        self.centre = np.array(['centre_x', 'centre_y'])
        self.CLASSES = np.append(self.classes, self.centre)

        #self.path_images = os.path.join(self.path, self.images)
        #self.path_etiquettes = os.path.join(self.path, self.etiquettes)

    def concat_classe_sous_classe(self, classe, sous_classe):

        if sous_classe != "":
            classe_sous_classe = str(classe) + "_" + str(sous_classe)
        else:
            classe_sous_classe = str(classe)
        return classe_sous_classe

    def concat_chemin_image(self, classe, sous_classe, nom_image=""):

        fichier_classe = "class_" + str(classe)

        if classe == 2 or classe == 3:
            fichier_sous_classe = "number_" + str(sous_classe)
        else:
            fichier_sous_classe = str(sous_classe)

        if fichier_sous_classe != "":
            chemin = os.path.join(self.path, self.images_etiquettes[0], fichier_classe, fichier_sous_classe)
        else:
            chemin = os.path.join(self.path, self.images_etiquettes[0], fichier_classe)

        if nom_image != "":
            chemin = os.path.join(chemin, nom_image)

        return chemin

    def reconstitution_nom_image(self, numero, classe_sous_classe):

        nom = "img_" + str(classe_sous_classe) + "_" + str(numero) + ".png"
        return nom

    def reconstitution_nom_etiquette(self, classe_sous_classe):

        nom = "label_" + str(classe_sous_classe) + ".csv"
        return nom

    #Fonction renvoyant un array de la forme [0, 0, 1, 0, ..] en fonction de l'appartenance de l'element à une sous-classe
    def comparer_element_classe(self, element):

        resultat = np.zeros(len(self.etiquettes))
        for i in range(len(self.etiquettes)):
            if self.etiquettes[i] == element:
                resultat[i] = 1
        return resultat

    #Constitution d'une etiquette propre à une image, avec centres normalisés (pour que les valeurs soient comprises entre 0 et 1)
    #Exemple : si l'image est classe1, rouge, le resultat sera [1, 0, 0, 0.465, 0.178] ( à changer pour le rpétraitement général)
    def constitution_etiquette(self, classe_sous_classe, cx, cy):

        etiquette_sous_classe = self.comparer_element_classe(classe_sous_classe)
        centre = np.array([cx/self.image_original_height, cy/self.image_original_width])
        #etiquette = np.append(etiquette_sous_classe, centre)
        #etiquette = (etiquette_sous_classe, centre)
        return etiquette_sous_classe, centre

    def _get_class(self, nom_image, classe, sous_classe): #sous_classe est un string

        etiquette = []
        classe_sous_classe = self.concat_classe_sous_classe(classe, sous_classe)

        if classe != 5:

            nom_etiquette = self.reconstitution_nom_etiquette(classe_sous_classe) #reconstitution du nom du fichier csv
            chemin = os.path.join(self.path, self.images_etiquettes[1], nom_etiquette) #reconstitution du chemin menant au fichier csv
            labels = open_file.lire_csv(chemin, classe, sous_classe) #recuperation de toutes les etiquettes
            #Recuperation de la bonne etiquette, sans le nom de l'image
            for label in labels:
                if label[0] == nom_image:
                    #etiquette = self.constitution_etiquette(classe_sous_classe, label[3], label[4])
                    etiquette_classe, etiquette_centre = self.constitution_etiquette(classe_sous_classe, label[3], label[4])
        else:
            #etiquette = self.constitution_etiquette(classe_sous_classe, 0, 0)
            etiquette_classe, etiquette_centre = self.constitution_etiquette(classe_sous_classe, 0, 0)

        #tenseur = tf.convert_to_tensor(etiquette) #conversion en tenseur
        return etiquette_classe, etiquette_centre
        #pat_splited = tf.strings.split(path, os.path.sep)
        #return pat_splited[–2] == CLASS_NAMES

    def _load_image(self, numero_image, classe, sous_classe):

        classe_sous_classe = self.concat_classe_sous_classe(classe, sous_classe)
        nom_image = self.reconstitution_nom_image(numero_image, classe_sous_classe)
        chemin = self.concat_chemin_image(classe, sous_classe, nom_image=nom_image)

        img_array = cv2.imread(chemin)  # , cv2.IMREAD_GRAYSCALE)  # convert to array
        img_array = img_array / 255.0
        self.image_original_height = img_array.shape[0]
        self.image_original_width = img_array.shape[1]
        #image = tf.io.read_file(path)
        #image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(img_array, tf.float32)
        image_redim = tf.image.resize(image, [self.image_height, self.image_width])
        return image_redim

    def load_single_data(self, numero_image, classe, sous_classe):

        classe_sous_classe = self.concat_classe_sous_classe(classe, sous_classe)
        nom_image = self.reconstitution_nom_image(numero_image, classe_sous_classe)
        image = self._load_image(numero_image, classe, sous_classe)
        label_classe, label_centre = self._get_class(nom_image, classe, sous_classe)
        return image, label_classe, label_centre

    def _load_labeled_data(self):

        images = []
        labels_classe = []
        labels_centre = []
        classe = 0
        datasets = []

        for cl in self.arborescence:
            classe += 1
            for sous_classe in cl:
                chemin = self.concat_chemin_image(classe, sous_classe)
                nbre_image = len(os.listdir(chemin))

                for i in range(1, nbre_image+1, 1):
                    #single_data = self.load_single_data(i, "1", sous_classe)
                    image, label_classe, label_centre = self.load_single_data(i, classe, sous_classe)
                    #data.append(single_data)
                    images.append(image)
                    labels_classe.append(label_classe)
                    labels_centre.append(label_centre)

                images_tensor = tf.convert_to_tensor(images)
                labels_classe_tensor = tf.convert_to_tensor(labels_classe)
                labels_centre_tensor = tf.convert_to_tensor(labels_centre)

                datasets.append(tf.data.Dataset.from_tensor_slices((images_tensor, (labels_classe_tensor, labels_centre_tensor))))

                images = []
                labels_classe = []
                labels_centre = []

        #print(images_tensor)
        #print(labels_tensor)

        dataset_final = datasets[0]
        for i in range(1, len(datasets), 1):
            dataset_final = dataset_final.concatenate(datasets[i])

        return dataset_final
        #return images_tensor, labels_tensor

        #print(data[0])
        #tenseur = tf.convert_to_tensor(data)
        #return data

    def get_dataset_partitions_tf(self, ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1):

        assert (train_split + test_split + val_split) == 1

        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)

        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)

        return train_ds, val_ds, test_ds

    def load_process(self, shuffle_size = 1000):

        #images, labels = self._load_labeled_data()

        self.loaded_dataset = self._load_labeled_data()
        #self.loaded_dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        #self.loaded_dataset = self.dataset.map(self._load_labeled_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.loaded_dataset = self.loaded_dataset.cache()

        # Shuffle data and create batches
        self.loaded_dataset = self.loaded_dataset.shuffle(buffer_size=shuffle_size)

        self.train, self.val, self.test = self.get_dataset_partitions_tf(self.loaded_dataset, 957)

        self.train = self.train.repeat()
        self.train = self.train.batch(self.batch_size)

        # Make dataset fetch batches in the background during the training of the model.
        self.train = self.train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


        #print(self.loaded_dataset.__len__())
        #print(tf.data.experimental.cardinality(self.loaded_dataset))

    def load_process_simple(self):

        self.loaded_dataset = self._load_labeled_data()

        self.loaded_dataset = self.loaded_dataset.cache()

        # Shuffle data and create batches
        #self.loaded_dataset = self.loaded_dataset.shuffle(buffer_size=shuffle_size)

        self.train, self.valid, self.test = self.get_dataset_partitions_tf(self.loaded_dataset, 957)

        #self.train = self.train.repeat(3)
        #self.train = self.train.batch(self.batch_size)

    def get_batch(self):
        return next(iter(self.train))


if __name__ == "__main__":

    dataProcessor = DataSetCreator(32, 300, 500)
    dataProcessor.load_process()

    image_batch, label_batch = dataProcessor.get_batch()
    print(type(image_batch))
    print(type(label_batch))
    print(image_batch)
    print(label_batch)
    #image_batch = dataProcessor.get_batch()