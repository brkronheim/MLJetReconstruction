""" 
Training code for the jet correction network. This network is used to correct a jet's 4-vector after the full algorithm has been run
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #supress tensorflow warning messages
warnings.filterwarnings("ignore", category=FutureWarning) 
import os
import random as rn
import datetime
import click



import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from math import pi
from metrics import metrics




@click.command()
@click.option('--hidden', default=5, help='Number of hidden layers')
@click.option('--width', default=50, help='Width of the hidden layers')
@click.option('--alpha', default=0.2, help='Slope for leaky relu')
@click.option('--initialLR', default=0.01, help='initial learning rate')
@click.option('--cycles', default=4, help='Number of cylces to train for')
@click.option('--epochs', default=100, help='Number of epochs in a cylce')
@click.option('--patience', default=20, help='Number of epochs with no improvement before ending')
@click.option('--dataName', default="gitJetData.npy", help='Name of input data file')
@click.option('--networkName', default="gitCorrection1", help='Name of network')


def main(hidden, width, alpha, initiallr, cycles, epochs, patience, dataname, networkname):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(42)
    rn.seed(12345)
    tf.random.set_seed(3)

    #Define some parameters for the network
    inputFileName = dataname
    outputModelName = networkname
    initialLR = initiallr


    dataInitial=np.load(inputFileName).T
    for x in [0,3,12,15]:
        dataInitial[x] = np.log(tf.math.maximum(dataInitial[x],0.01))

    meanAlgPt=np.mean(dataInitial[0])
    meanAlgEta=np.mean(dataInitial[1])
    meanAlgPhi=0
    meanAlgE=np.mean(dataInitial[3])

    stdAlgPt=np.std(dataInitial[0])
    stdAlgEta=np.std(dataInitial[1])
    stdAlgPhi=1
    stdAlgE=np.std(dataInitial[3])

    meanGenPt=np.mean(dataInitial[12])
    meanGenEta=np.mean(dataInitial[13])
    meanGenPhi=0
    meanGenE=np.mean(dataInitial[15])

    stdGenPt=np.std(dataInitial[12])
    stdGenEta=np.std(dataInitial[13])
    stdGenPhi=1
    stdGenE=np.std(dataInitial[15])

    metricObject = metrics(meanGenPt, stdGenPt, meanGenEta, stdGenEta, meanGenPhi, stdGenPhi)

    #normalization
    for x in [0,1,3,12,13,15]:
        dataInitial[x] = (dataInitial[x]-np.mean(dataInitial[x]))/np.std(dataInitial[x])
        print()



    inputData=dataInitial[0:4].T
    outputData=dataInitial[12:].T

    #Randomly pick an 90-10 train-validation split
    trainIn, valIn, trainOut, valOut = train_test_split(inputData,
                                                        outputData,
                                                        test_size=1/9,
                                                        random_state=42)


    trainIn = trainIn.astype(np.float32)
    valIn = valIn.astype(np.float32)
    trainOut = trainOut.astype(np.float32)
    valOut =  valOut.astype(np.float32)




    tf.random.set_seed(1000) #Set seed

    inputDims=4
    outputDims=4


    #Start building the model.
    model = tf.keras.Sequential()


    model.add(
        tf.keras.layers.Dense(
            width,
            kernel_initializer="glorot_uniform",
            input_shape=(
                inputDims,
            )))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))

    for n in range(hidden - 1):
        model.add(
            tf.keras.layers.Dense(
                width,
                kernel_initializer="glorot_uniform"))
        model.add(tf.keras.layers.LeakyReLU(alpha=alpha))

    model.add(
        tf.keras.layers.Dense(
            outputDims,
            kernel_initializer="glorot_uniform",
            activation=None))
    callbackMetric="val_loss"

    callback = tf.keras.callbacks.EarlyStopping(
            monitor=callbackMetric, patience=patience, restore_best_weights=True)
    #Train for the desired number of cycles
    for x in range(cycles):
        model.compile(optimizer=tf.keras.optimizers.Adam(initialLR * (10**(-x)),
                      amsgrad=True),
                      loss=metricObject.custom_loss,
                      metrics=["mae", "mse",metricObject.pt_error, metricObject.eta_error, metricObject.phi_error, metricObject.e_error, metricObject.dPT, metricObject.dR])
        model.summary()

        model.fit(
            trainIn,
            trainOut,
            validation_data=(
                valIn,
                valOut),
            epochs=epochs,
            batch_size=32,
            #sample_weight=weights,
            callbacks=[callback])

    model.save(outputModelName)

if(__name__=="__main__"):
    main()
