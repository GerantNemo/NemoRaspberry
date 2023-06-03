#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pathlib
import matplotlib.pyplot as plt
import cv2
import csv
import pandas as pd
import numpy as np
import scipy
from DetectionImage.Pretraitement import pretraitement
import PyQt5
import tensorflow as tf

taille_x = 28
taille_y = 28
taille_z = 3


def preprocess(image, label):
    resized_image = tf.image.resize(image, [taille_x, taille_y])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

def modele2_entrainement():

    dataProcessor = pretraitement.DataSetCreator(32, 224, 224)
    dataProcessor.load_process_simple()

    n_classes = 13

    train_set = dataProcessor.train
    valid_set = dataProcessor.valid
    test_set = dataProcessor.valid

    batch_size = 5
    train_set = train_set.shuffle(1000)
    train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
    valid_set = valid_set.map(preprocess).batch(batch_size).prefetch(1)
    test_set = test_set.map(preprocess).batch(batch_size).prefetch(1)

    #print(train_set)

    #x_train_set, y_train_set = train_set
    #x_test_set, y_test_set = test_set

    #base_model = tf.keras.applications.xception.Xception(weights="imagenet",include_top=False)
    #avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    #class_output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
    #loc_output = tf.keras.layers.Dense(2)(avg)

    base_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, activation="relu", padding="same",
                            input_shape=[taille_x, taille_y, taille_z]),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        #tf.keras.layers.Dense(10, activation="softmax")
    ])

    #avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    class_output = tf.keras.layers.Dense(n_classes, activation="softmax")(base_model.output)
    loc_output = tf.keras.layers.Dense(2)(base_model.output)

    model = tf.keras.Model(inputs=base_model.input, outputs=[class_output, loc_output])

    model.summary()

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01)
    model.compile(loss=["sparse_categorical_crossentropy", "mse"], loss_weights=[0.8, 0.2] ,optimizer=optimizer, metrics=["accuracy"])
    history = model.fit(train_set, epochs=5, validation_data=valid_set)

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # Régler la plage verticale sur [0‐1]
    plt.show()

    Test = model.evaluate(test_set)


if __name__ == "__main__":

    modele2_entrainement()