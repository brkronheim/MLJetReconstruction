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
inputFileName = "mlDataNew2.npy"
outputModelName = "jet_correction_alg_long_fixed_phi2"

width=50
alpha=0.2
hidden=5
cycles=4
epochs=100
patience=20
initialLR = 0.01

#Define custom loss with the correct handling of phi, as well as some useful metrics


def dR(y_true, y_pred):
    
    meanPT = 210.54679049884757
    sdPT = 272.11917496982136
    meanEta = 0.006930566673335255
    sdEta = 1.6144690552693344
    meanPhi = 0
    sdPhi = 1

    
    
    truePT=tf.math.add(tf.math.multiply(y_true[:,0], sdPT), meanPT)
    trueEta=tf.math.add(tf.math.multiply(y_true[:,1], sdEta), meanEta)
    truePhi=tf.math.add(tf.math.multiply(y_true[:,1], sdPhi), meanPhi)
    
    predPT=tf.math.add(tf.math.multiply(y_pred[:,0], sdPT), meanPT)
    predEta=tf.math.add(tf.math.multiply(y_pred[:,1], sdEta), meanEta)
    predPhi=tf.math.add(tf.math.multiply(y_pred[:,1], sdPhi), meanPhi)
    

    raw_dphi = truePhi - predPhi;
    dphi = tf.where(tf.abs(raw_dphi)<pi, raw_dphi, raw_dphi-2*pi*tf.math.round(raw_dphi/2*pi))
    
    deta = trueEta - predEta;
    val=tf.math.pow(tf.math.pow(deta,2)+tf.math.pow(dphi,2),0.5)
    return(tf.reduce_mean(val))


    
def dPT(y_true, y_pred):
    meanPT = 210.54679049884757
    sdPT = 272.11917496982136
    
    
    truePT=tf.math.add(tf.math.multiply(y_true[:,0], sdPT), meanPT)
    
    predPT=tf.math.add(tf.math.multiply(y_pred[:,0], sdPT), meanPT)
    
    val=tf.math.divide(predPT,truePT)
    #print(val)
    return(tf.reduce_mean(val))



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

dataInitial=np.load(inputFileName).T

meanAlgPt=np.mean(dataInitial[0])
meanAlgEta=np.mean(dataInitial[1])
meanAlgPhi=np.mean(dataInitial[2])
meanAlgE=np.mean(dataInitial[3])

stdAlgPt=np.std(dataInitial[0])
stdAlgEta=np.std(dataInitial[1])
stdAlgPhi=np.std(dataInitial[2])
stdAlgE=np.std(dataInitial[3])

meanGenPt=np.mean(dataInitial[12])
meanGenEta=np.mean(dataInitial[13])
meanGenPhi=np.mean(dataInitial[14])
meanGenE=np.mean(dataInitial[15])

stdGenPt=np.std(dataInitial[12])
stdGenEta=np.std(dataInitial[13])
stdGenPhi=np.std(dataInitial[14])
stdGenE=np.std(dataInitial[15])

print()
print(meanGenPt)
print(meanGenEta)
print(meanGenPhi)
print(meanGenE)
print()
print(stdGenPt)
print(stdGenEta)
print(stdGenPhi)
print(stdGenE)
print()
#normalization
for x in [0,1,3,12,13,15]:
    print(np.mean(dataInitial[x]))
    print(np.std(dataInitial[x]))
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
                  loss=custom_loss,
                  metrics=["mae", "mse",dPT, dR])
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
        callbacks=[callback,tensorboard_callback])

model.save("jet_correction_alg_long_fixed_phi2")
