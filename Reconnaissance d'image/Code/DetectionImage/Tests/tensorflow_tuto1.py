#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def tuto1():

    print("TensorFlow version:", tf.__version__)

    #Chargement du jeu de donnees mnist et conversion des donnees d'entiers en nombre à virgule flottante
    fashion_mnist = tf.keras.datasets.mnist

    (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
    x_valid, x_train = x_train_full[:5000] / 255.0, x_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    #Creation d'un modele equential en empilant des couches
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

    model.summary()

    hidden1 = model.layers[1]
    weights, biases = hidden1.get_weights()
    print(weights, biases)
    print(weights.shape)
    print(biases.shape)

    #Le modele renvoie un vecteur de score logits pour chaque exemple
    predictions = model(x_train[:1]).numpy()
    print(predictions)

    #Conversion des logits en probabilités pour chaque classe
    print(tf.nn.softmax(predictions).numpy())

    #Définition d'une fonction de perte
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    #Le modèle non formé donne des probabilités proches du hasard (2,3 environ)
    print(loss_fn(y_train[:1], predictions).numpy())

    #Configuration et compilation du modèle
    model.compile(optimizer='sgd',
                  loss=loss_fn,
                  metrics=['accuracy'])

    #Entrainement du modèle
    history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))

    #Evaluation du modèle selon un Test set
    model.evaluate(x_test, y_test, verbose=2)

    #Enveloppement du modèle et attachement du softmax pour le renvoi d'une probabilité
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    print(probability_model(x_test[:5]))

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1) #Réglage de la plage verticale sur 0-1
    plt.show()


if __name__ == "__main__":

    tuto1()