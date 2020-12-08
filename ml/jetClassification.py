import os
import random as rn
import datetime

from math import pi

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(3)

#Define some parameters for the network
inputFileName = "mlDataNoDupsClean.npy"
outputModelName = "classification_verticies_seed_no_weights_no_dups"

width=50
alpha=0.2
hidden=5
cycles=2
epochs=50
patience=5
initialLR = 0.001

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
    correct=0
    weight=1
    #Test if matched jet
    if(dataInitial[x,1] !=0 or dataInitial[x,2]!=0 or dataInitial[x,3]!=0 or dataInitial[x,4]!=0 or dataInitial[x,5]!=0 or dataInitial[x,0]!=0):
        weight = 5.14
        correct=1
    #Test if start of a new jet    
    if(dataInitial[x,0]==1):
        
        #Set initial jet as 0 jet
        jetPX = 0.0 
        jetPY = 0.0
        jetPZ = 0.0
        jetE = 0.0 
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
    temp.append(firstPT)
    temp.append(firstEta)
    temp.append(firstPhi)
    temp.append(firstE)
    temp.append(firstVx)
    temp.append(firstVy)
    temp.append(firstVz)
    temp.append(jetPT)
    temp.append(jetEta)
    temp.append(jetPhi)
    temp.append(jetE)
    temp.append(particlePT)
    temp.append(particleEta)
    temp.append(particlePhi)
    temp.append(dataInitial[x,9])
    temp.append(dataInitial[x,10])
    temp.append(dataInitial[x,11])
    temp.append(dataInitial[x,12])
    temp= temp + newPDGID[dataInitial[x,13]]
    jetPX += dataInitial[x,1]
    jetPY += dataInitial[x,2]
    jetPZ += dataInitial[x,3]
    jetE += dataInitial[x,4]
    temp.append(correct)
    data.append(np.array(temp))
    

data=np.array(data).T
print(data.shape)

meanX=np.mean(data[7])
meanY=np.mean(data[8])
meanZ=np.mean(data[9])
meanE=np.mean(data[10])

meanVx=np.mean(data[15])
meanVy=np.mean(data[16])
meanVz=np.mean(data[17])

sdX=np.std(data[7])
sdY=np.std(data[8])
sdZ=np.std(data[9])
sdE=np.std(data[10])

sdVx=np.std(data[15])
sdVy=np.std(data[16])
sdVz=np.std(data[17])

data[0]=(data[0]-meanX)/sdX
data[1]=(data[1]-meanY)/sdY
data[2]=(data[2]-meanZ)/sdZ
data[3]=(data[3]-meanE)/sdE
data[4]=(data[4]-meanVx)/sdVx
data[5]=(data[5]-meanVy)/sdVy
data[6]=(data[6]-meanVz)/sdVz

data[7]=(data[7]-meanX)/sdX
data[8]=(data[8]-meanY)/sdY
data[9]=(data[9]-meanZ)/sdZ
data[10]=(data[10]-meanE)/sdE
data[11]=(data[11]-meanX)/sdX
data[12]=(data[12]-meanY)/sdY
data[13]=(data[13]-meanZ)/sdZ
data[14]=(data[14]-meanE)/sdE

data[15]=(data[15]-meanVx)/sdVx
data[16]=(data[16]-meanVy)/sdVy
data[17]=(data[17]-meanVz)/sdVz



inputData=data[0:29].T
outputData=data[29:31].T

#Randomly pick an 90-10 train-validation split
trainIn, valIn, trainOut, valOut = train_test_split(inputData,
                                                    outputData,
                                                    test_size=1/9,
                                                    random_state=42)

trainIn = trainIn.astype(np.float32)
valIn = valIn.astype(np.float32)
trainOut = trainOut.astype(np.float32)
valOut =  valOut.astype(np.float32)

weights=trainOut.T[1]
trainOut=trainOut.T[0]
valOut=valOut.T[0]


inputDims=29
outputDims=1

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
        activation="sigmoid"))
callbackMetric="val_loss"

callback = tf.keras.callbacks.EarlyStopping(
        monitor=callbackMetric, patience=patience, restore_best_weights=True)
#Train for the desired number of cycles
for x in range(cycles):
    model.compile(optimizer=tf.keras.optimizers.Adam(initialLR * (10**(-x)),
                  amsgrad=True),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    model.summary()
    
    model.fit(
        trainIn,
        trainOut,
        validation_data=(
            valIn,
            valOut),
        epochs=epochs[x],
        batch_size=64,
        callbacks=[callback,tensorboard_callback])

model.save(outputModelName)
