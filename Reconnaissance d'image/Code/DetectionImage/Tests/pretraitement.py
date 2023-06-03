#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
#import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import csv
import pandas as p

def print_label_mnist():

    print("TensorFlow version:", tf.__version__)

    #Chargement du jeu de donnees mnist et conversion des donnees d'entiers en nombre à virgule flottante
    fashion_mnist = tf.keras.datasets.mnist

    (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
    x_valid, x_train = x_train_full[:5000] / 255.0, x_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    plt.figure()
    plt.imshow(x_train_full[0])
    plt.grid(False)
    plt.show()

    print(y_train_full)

def print_label_banque_min():

    path = "D:\\Aymeric\\Banque_images_reduites"
    images = "images"
    etiquettes = "etiquettes"

    path_images = os.path.join(path, images)
    path_etiquettes = os.path.join(path, etiquettes)

    for img in os.listdir(path_images):  # iterate over each image per dogs and cats
        print(os.path.join(path_images,img))
        img_array = cv2.imread(os.path.join(path_images,img))#, cv2.IMREAD_GRAYSCALE)  # convert to array
        #cv2.imshow("Image", img_array)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        print(img_array)
        #time.sleep(5)
        #plt.imshow(img_array)  # graph it
        #plt.show()  # display!

    with open(os.path.join(path_etiquettes, "label.csv")) as file:
        etiquettes = p.read_csv(file, index_col="region_attributes")
        row = etiquettes.iloc[1]
        print(row)
        print(etiquettes)
        print(type(etiquettes))
        for etiquette in etiquettes:
            print(etiquette)
        #cpt = 0
        #for etiquette in etiquettes:
        #    if cpt > 0:
        #        print(etiquette[0])
        #    cpt =+ 1



if __name__ == "__main__":

    print_label_banque_min()