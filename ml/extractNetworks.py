""" 
Converts a text file output by the C++ code into a .npy file where the particles in a jet are sorted by pT
"""

import click
import numpy as np
import matplotlib.pyplot as plt
import os
import random as rn
import datetime

from math import pi

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


@click.command()
@click.option('--filename', default="gitParticleData.npy", help='name of data file for normalization info')
@click.option('--regression', default="gitRegression1", help='name of fregression network')
@click.option('--classification', default="gitClassification1", help='name of classification network')

def main(filename, regression, classification):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    extractRegression(filename, regression)
    extractClassification(filename, classification)


    
    
    
def extractRegression(filename, regression):
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(42)
    rn.seed(12345)
    tf.random.set_seed(3)


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

    dataInitial=np.load(filename)
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
    


    normInfo1 = np.zeros((29, 2))

    for x in range(18, 29):
        normInfo1[x,1] = 1

    for x in [4, 15]:
        normInfo1[x,0] = meanVx
        normInfo1[x,1] = sdVx

    for x in [5, 16]:
        normInfo1[x,0] = meanVy
        normInfo1[x,1] = sdVy

    for x in [6, 17]:
        normInfo1[x,0] = meanVz
        normInfo1[x,1] = sdVz


    for x in [0,7,11]:
        normInfo1[x,0] = meanPt
        normInfo1[x,1] = sdPt

    for x in [1,8,12]:
        normInfo1[x,0] = meanEta
        normInfo1[x,1] = sdEta

    for x in [2,9,13]:
        normInfo1[x,0] = meanPhi
        normInfo1[x,1] = sdPhi

    for x in [3,10,14]:
        normInfo1[x,0] = meanE
        normInfo1[x,1] = sdE

    normInfo2 = np.zeros((4,2))
    for x in [0]:
        normInfo2[x,0] = meanPt
        normInfo2[x,1] = sdPt

    for x in [1]:
        normInfo2[x,0] = meanEta
        normInfo2[x,1] = sdEta

    for x in [2]:
        normInfo2[x,0] = meanPhi
        normInfo2[x,1] = sdPhi

    for x in [3]:
        normInfo2[x,0] = meanE
        normInfo2[x,1] = sdE

    model = tf.keras.models.load_model(regression, compile=False)#,custom_objects={"custom_loss": custom_loss})
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01,
                      amsgrad=True),loss=tf.keras.losses.MeanSquaredError(),
                      metrics=["mae", "mse"])

    #Extract weights and biases
    weights = []
    biases = []
    for layer in model.layers:
        weightBias = layer.get_weights()
        if(len(weightBias) == 2):
            weights.append(weightBias[0].T)
            bias = weightBias[1]
            bias = np.reshape(bias, (len(bias), 1))
            biases.append(bias)

    #Save weights and biases to a text file
    for x in range(len(weights)):
        np.savetxt("regression"+"/weights_"+str(x)+".txt",
                   weights[x], delimiter=",")

    for x in range(len(biases)):
        np.savetxt("regression"+"/biases_"+str(x)+".txt",
                   biases[x], delimiter=",")

    #Save normalization information to a text file
    np.savetxt("regression"+"/normInfo1.txt",
               normInfo1, delimiter=",")

    #Save normalization information to a text file
    np.savetxt("regression"+"/normInfo2.txt",
               normInfo2, delimiter=",")

def extractClassification(filename, classification):
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(42)
    rn.seed(12345)
    tf.random.set_seed(3)
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

    dataInitial=np.load(filename)
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
        correct=0
        weight=1
        #Test if matched jet
        if(dataInitial[x,1] !=0 or dataInitial[x,2]!=0 or dataInitial[x,3]!=0 or dataInitial[x,4]!=0 or dataInitial[x,5]!=0 or dataInitial[x,0]!=0):
            weight = 5.14
            correct=1
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
        temp.append(correct)
        data.append(np.array(temp))


    data=np.array(data).T


    meanPt=np.mean(data[7])
    meanEta=np.mean(data[8])
    meanPhi=np.mean(data[9])
    meanE=np.mean(data[10])

    meanVx=np.mean(data[15])
    meanVy=np.mean(data[16])
    meanVz=np.mean(data[17])

    sdPt=np.std(data[7])
    sdEta=np.std(data[8])
    sdPhi=np.std(data[9])
    sdE=np.std(data[10])

    sdVx=np.std(data[15])
    sdVy=np.std(data[16])
    sdVz=np.std(data[17])

    normInfo1 = np.zeros((29, 2))

    for x in range(18, 29):
        normInfo1[x,1] = 1

    for x in [4, 15]:
        normInfo1[x,0] = meanVx
        normInfo1[x,1] = sdVx

    for x in [5, 16]:
        normInfo1[x,0] = meanVy
        normInfo1[x,1] = sdVy

    for x in [6, 17]:
        normInfo1[x,0] = meanVz
        normInfo1[x,1] = sdVz


    for x in [0,7,11]:
        normInfo1[x,0] = meanPt
        normInfo1[x,1] = sdPt

    for x in [1,8,12]:
        normInfo1[x,0] = meanEta
        normInfo1[x,1] = sdEta

    for x in [2,9,13]:
        normInfo1[x,0] = meanPhi
        normInfo1[x,1] = sdPhi

    for x in [3,10,14]:
        normInfo1[x,0] = meanE
        normInfo1[x,1] = sdE

    normInfo2 = np.zeros((4,2))
    for x in [0]:
        normInfo2[x,0] = meanPt
        normInfo2[x,1] = sdPt

    for x in [1]:
        normInfo2[x,0] = meanEta
        normInfo2[x,1] = sdEta

    for x in [2]:
        normInfo2[x,0] = meanPhi
        normInfo2[x,1] = sdPhi

    for x in [3]:
        normInfo2[x,0] = meanE
        normInfo2[x,1] = sdE

    model = tf.keras.models.load_model(classification, compile=False)#,custom_objects={"custom_loss": custom_loss})
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01,
                      amsgrad=True),loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    #Extract weights and biases
    weights = []
    biases = []
    for layer in model.layers:
        weightBias = layer.get_weights()
        if(len(weightBias) == 2):
            weights.append(weightBias[0].T)
            bias = weightBias[1]
            bias = np.reshape(bias, (len(bias), 1))
            biases.append(bias)

    #Save weights and biases to a text file
    for x in range(len(weights)):
        np.savetxt("classification"+"/weights_"+str(x)+".txt",
                   weights[x], delimiter=",")

    for x in range(len(biases)):
        np.savetxt("classification"+"/biases_"+str(x)+".txt",
                   biases[x], delimiter=",")

    #Save normalization information to a text file
    np.savetxt("classification"+"/normInfo1.txt",
               normInfo1, delimiter=",")


if(__name__=="__main__"):
    main()
