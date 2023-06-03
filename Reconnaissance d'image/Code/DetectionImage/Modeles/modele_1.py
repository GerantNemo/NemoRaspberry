#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import csv
import pandas as pd
import numpy as np
import scipy
from DetectionImage.Pretraitement import pretraitement
import time

root_logdir = "D:\\Aymeric\\Documents divers et variés\\Programmation\\Python\\DeepLearning\\Logs_detection_image"

def get_run_logdir(step):
    if step == 1:
        run_id = time.strftime("modele1_step1_run_%Y_%m_%d-%H_%M_%S")
    else:
        run_id = time.strftime("modele1_step2_run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

def modele1_entrainement():

    dataProcessor = pretraitement.DataSetCreator(32, 224, 224)
    dataProcessor.load_process_simple()

    n_classes = 13

    train_set = dataProcessor.train
    valid_set = dataProcessor.valid
    test_set = dataProcessor.valid

    batch_size = 32
    train_set = train_set.shuffle(1000)
    train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
    valid_set = valid_set.map(preprocess).batch(batch_size).prefetch(1)
    test_set = test_set.map(preprocess).batch(batch_size).prefetch(1)

    base_model = tf.keras.applications.xception.Xception(weights="imagenet",include_top=False)
    avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    class_output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
    loc_output = tf.keras.layers.Dense(2)(avg)

    model = tf.keras.Model(inputs=base_model.input, outputs=[class_output, loc_output])

    for layer in base_model.layers:
        layer.trainable = False

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01)
    model.compile(loss=["sparse_categorical_crossentropy", "mse"], loss_weights=[0.8, 0.2] ,optimizer=optimizer, metrics=["accuracy"])

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("modele1_step1.h5", save_best_only=True)
    run_logdir = get_run_logdir(1)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
    history = model.fit(train_set, epochs=5, validation_data=valid_set, callbacks=[tensorboard_cb, checkpoint_cb])
    model = tf.keras.models.load_model("modele1_step1.h5")  # Revenir au meilleur modèle

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # Régler la plage verticale sur [0‐1]
    plt.show()


    for layer in base_model.layers:
        layer.trainable = False

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01)
    model.compile(loss=["sparse_categorical_crossentropy", "mse"], loss_weights=[0.8, 0.2] ,optimizer=optimizer, metrics=["accuracy"])

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("modele1_step2.h5", save_best_only=True)
    run_logdir = get_run_logdir(2)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
    history = model.fit(train_set, epochs=5, validation_data=valid_set, callbacks=[tensorboard_cb, checkpoint_cb])
    model = tf.keras.models.load_model("modele1_step2.h5")  # Revenir au meilleur modèle
    #model.save("modele1.h5")

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # Régler la plage verticale sur [0‐1]
    plt.show()

    Test = model.evaluate(test_set)


if __name__ == "__main__":

    modele1_entrainement()