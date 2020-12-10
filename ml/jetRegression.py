""" 
Training code for the jet regression network. This network is used to update a jet's 4-vector when given
a new potential particle.
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
@click.option('--initialLR', default=0.001, help='initial learning rate')
@click.option('--cycles', default=3, help='Number of cylces to train for')
@click.option('--epochs', default=100, help='Number of epochs in a cylce')
@click.option('--patience', default=20, help='Number of epochs with no improvement before ending')
@click.option('--dataName', default="gitParticleDataClean.npy", help='Name of input data file')
@click.option('--networkName', default="gitRegression1", help='Name of network')


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

    #Dicitionary for one-hot encoding
    newPDGID = {-11.0:[1,0,0,0,0,0,0,0,0,0,0], 
                -13.0:[0,1,0,0,0,0,0,0,0,0,0], 
                -211:[0,0,1,0,0,0,0,0,0,0,0], 
                22.0:[0,0,0,1,0,0,0,0,0,0,0], 
                1.0:[0,0,0,0,1,0,0,0,0,0,0], 
                0.0:[0,0,0,0,0,1,0,0,0,0,0], 
                2.0:[0,0,0,0,0,0,1,0,0,0,0], 
                130.0:[0,0,0,0,0,0,0,1,0,0,0], 
                211.0:[0,0,0,0,0,0,0,0,1,0,0], 
                13.0:[0,0,0,0,0,0,0,0,0,1,0], 
                11.0:[0,0,0,0,0,0,0,0,0,0,1]};

    dataInitial=np.load(inputFileName)
    data=[]


    jetPX = 0.0 
    jetPY = 0.0
    jetPZ = 0.0
    jetE = 0.0
    firstPt = 0.0
    firstEta = 0.0
    firstPhi = 0.0
    firstE = 0.0
    firstVx=0
    firstVy=0
    firstVz=0
    currentX=0.0
    jetSize=[]
    for x in range(int(len(dataInitial))):
        temp=[]
        #Test if a matched jet
        if(dataInitial[x,1] !=0 or dataInitial[x,2]!=0 or dataInitial[x,3]!=0 or dataInitial[x,4]!=0 or dataInitial[x,5]!=0 or dataInitial[x,0]!=0):
            #Test if start of a new jet
            if(dataInitial[x,0]==1):

                #Set initial jet as 0 jet
                jetPX = 0.01 
                jetPY = 0.0
                jetPZ = 0.0
                jetE = 0.01 
                jetSize.append(x-currentX)
                currentX=x

                #Get info on seed jet
                firstVx=dataInitial[x,10]
                firstVy=dataInitial[x,11]
                firstVz=dataInitial[x,12]

                firstPT = (dataInitial[x,6]**2 + dataInitial[x,7]**2)**0.5
                firstPhi = 0.0
                firstEta = 0.0
                if(firstPT != 0):
                    firstPhi = np.arcsin(dataInitial[x,7]/firstPT)
                    firstEta = np.arcsinh(dataInitial[x,8]/firstPT)
                firstE = dataInitial[x,9]

            #Get information for new particle
            jetPT = (jetPX**2 + jetPY**2)**0.5
            jetPhi = 0.0
            jetEta = 0.0
            if(jetPT!=0):
                jetPhi = np.arcsin(jetPY/jetPT)
                jetEta = np.arcsinh(jetPZ/jetPT)
            particlePT = (dataInitial[x,6]**2 + dataInitial[x,7]**2)**0.5
            particlePhi = 0.0
            particleEta = 0.0
            if(particlePT != 0):
                particlePhi = np.arcsin(dataInitial[x,7]/particlePT)
                particleEta = np.arcsinh(dataInitial[x,8]/particlePT)
            #Add new entry to dataset
            temp.append(np.log(firstPT))
            temp.append(firstEta)
            temp.append(firstPhi)
            temp.append(np.log(firstE))
            temp.append(np.cbrt(firstVx))
            temp.append(np.cbrt(firstVy))
            temp.append(np.cbrt(firstVz))
            temp.append(np.log(jetPT))
            temp.append(jetEta)
            temp.append(jetPhi)
            temp.append(np.log(jetE))
            temp.append(np.log(particlePT))
            temp.append(particleEta)
            temp.append(particlePhi)
            temp.append(np.log(dataInitial[x,9]))
            temp.append(np.cbrt(dataInitial[x,10]))
            temp.append(np.cbrt(dataInitial[x,11]))
            temp.append(np.cbrt(dataInitial[x,12]))
            temp= temp + newPDGID[dataInitial[x,13]]
            jetPX += dataInitial[x,1]
            jetPY += dataInitial[x,2]
            jetPZ += dataInitial[x,3]
            jetE += dataInitial[x,4]
            jetPT = (jetPX**2 + jetPY**2)**0.5
            jetPhi = 0.0
            jetEta = 0.0
            if(jetPT!=0):
                jetPhi = np.arcsin(jetPY/jetPT)
                jetEta = np.arcsinh(jetPZ/jetPT)
            temp.append(np.log(jetPT))
            temp.append(jetEta)
            temp.append(jetPhi)
            temp.append(np.log(jetE))
            data.append(np.array(temp))

    data=np.array(data).T

    meanPt=np.mean(data[7])
    meanEta=np.mean(data[8])
    meanPhi=0
    meanE=np.mean(data[10])

    meanVx=np.mean(data[15])
    meanVy=np.mean(data[16])
    meanVz=np.mean(data[17])

    sdPt=np.std(data[7])
    sdEta=np.std(data[8])
    sdPhi=1
    sdE=np.std(data[10])

    sdVx=np.std(data[15])
    sdVy=np.std(data[16])
    sdVz=np.std(data[17])

    data[0]=(data[0]-meanPt)/sdPt
    data[1]=(data[1]-meanEta)/sdEta
    data[2]=(data[2]-meanPhi)/sdPhi
    data[3]=(data[3]-meanE)/sdE
    data[4]=(data[4]-meanVx)/sdVx
    data[5]=(data[5]-meanVy)/sdVy
    data[6]=(data[6]-meanVz)/sdVz

    data[7]=(data[7]-meanPt)/sdPt
    data[8]=(data[8]-meanEta)/sdEta
    data[9]=(data[9]-meanPhi)/sdPhi
    data[10]=(data[10]-meanE)/sdE
    data[11]=(data[11]-meanPt)/sdPt
    data[12]=(data[12]-meanEta)/sdEta
    data[13]=(data[13]-meanPhi)/sdPhi
    data[14]=(data[14]-meanE)/sdE

    data[15]=(data[15]-meanVx)/sdVx
    data[16]=(data[16]-meanVy)/sdVy
    data[17]=(data[17]-meanVz)/sdVz

    data[-4]=(data[-4]-meanPt)/sdPt
    data[-3]=(data[-3]-meanEta)/sdEta
    data[-2]=(data[-2]-meanPhi)/sdPhi
    data[-1]=(data[-1]-meanE)/sdE

    


    inputData=data[0:29].T
    outputData=data[29:].T

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

    inputDims=29
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

    metricObject = metrics(meanPt, sdPt, meanEta, sdEta, meanPhi, sdPhi)
    
    callback = tf.keras.callbacks.EarlyStopping(
            monitor=callbackMetric, patience=patience, restore_best_weights=True)
    #Train for the desired number of cycles
    for x in range(cycles):
        model.compile(optimizer=tf.keras.optimizers.Adam(initialLR * (10**(-x)),
                      amsgrad=True),
                      loss=metricObject.custom_loss,
                      metrics=[metricObject.pt_error, metricObject.eta_error, metricObject.phi_error, metricObject.e_error,metricObject.dPT, metricObject.dR])
        model.summary()

        model.fit(
            trainIn,
            trainOut,
            validation_data=(
                valIn,
                valOut),
            epochs=epochs,
            batch_size=64,
            callbacks=[callback])

    model.save(outputModelName)

if(__name__=="__main__"):
    main()
