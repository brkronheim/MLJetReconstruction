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
outputModelName = "jet_regression_big_fixed_phi_no_dups2

width = 50
alpha = 0.2
hidden = 5
cycles = 3
epochs = 100
patience = 20
initialLR = 0.001

#Define custom loss with the correct handling of phi, as well as individual metrics for each parameter
def custom_loss(y_true, y_pred):
    val = tf.minimum(tf.math.pow(tf.math.subtract(y_true[:,2], y_pred[:,2]),2),tf.math.pow(tf.math.subtract(y_true[:,2], y_pred[:,2]+2*pi),2))
    val = tf.minimum(val,tf.math.pow(tf.math.subtract(y_true[:,2], y_pred[:,2]-2*pi),2))
    val1 = tf.reduce_mean(val)
    val2 = tf.reduce_mean(tf.math.pow(tf.math.subtract(y_true[:,0], y_pred[:,0]),2))
    val3 = tf.reduce_mean(tf.math.pow(tf.math.subtract(y_true[:,1], y_pred[:,1]),2))
    val4 = tf.reduce_mean(tf.math.pow(tf.math.subtract(y_true[:,3], y_pred[:,3]),2))

    return([val1, val2, val3, val4])

def pt_error(y_true, y_pred):
    val = tf.reduce_mean(tf.math.pow(tf.math.subtract(y_true[:,0], y_pred[:,0]),2))
    
    return(val)

def eta_error(y_true, y_pred):
    val = tf.reduce_mean(tf.math.pow(tf.math.subtract(y_true[:,1], y_pred[:,1]),2))
    
    return(val)

def phi_error(y_true, y_pred):
    val = tf.minimum(tf.math.pow(tf.math.subtract(y_true[:,2], y_pred[:,2]),2),tf.math.pow(tf.math.subtract(y_true[:,2], y_pred[:,2]+2*pi),2))
    val = tf.minimum(val,tf.math.pow(tf.math.subtract(y_true[:,2], y_pred[:,2]-2*pi),2))
    val = tf.reduce_mean(val)
    return(val)

def e_error(y_true, y_pred):
    val = tf.reduce_mean(tf.math.pow(tf.math.subtract(y_true[:,3], y_pred[:,3]),2))
    
    return(val)

def dR(y_true, y_pred):
    
    meanPT = 215.37727054583502
    sdPT = 274.2569580648554
    meanEta = 0.0020797322291117477
    sdEta = 1.235316876252943
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
    meanPT = 215.37727054583502
    sdPT = 274.2569580648554
    
    truePT=tf.math.add(tf.math.multiply(y_true[:,0], sdPT), meanPT)
    
    predPT=tf.math.add(tf.math.multiply(y_pred[:,0], sdPT), meanPT)
    
    val=tf.math.divide(predPT,truePT)
    
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
        jetPT = (jetPX**2 + jetPY**2)**0.5
        jetPhi = 0.0
        jetEta = 0.0
        if(jetPT!=0):
            jetPhi = np.arcsin(jetPY/jetPT)
            jetEta = np.arcsinh(jetPZ/jetPT)
        temp.append(jetPT)
        temp.append(jetEta)
        temp.append(jetPhi)
        temp.append(jetE)
        data.append(np.array(temp))
  
data=np.array(data).T

meanPt=np.mean(data[7])
meanEta=np.mean(data[8])
meanPhi=0#np.mean(data[9])
meanE=np.mean(data[10])

meanVx=np.mean(data[15])
meanVy=np.mean(data[16])
meanVz=np.mean(data[17])

sdPt=np.std(data[7])
sdEta=np.std(data[8])
sdPhi=1#np.std(data[9])
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

#Print normalization info
print(meanPt)
print(sdPt)
print(meanEta)
print(sdEta)
print(meanPhi)
print(sdPhi)
print(meanE)
print(sdE)
print(meanVx)
print(sdVx)
print(meanVy)
print(sdVy)
print(meanVz)
print(sdVz)




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

callback = tf.keras.callbacks.EarlyStopping(
        monitor=callbackMetric, patience=patience, restore_best_weights=True)
#Train for the desired number of cycles
for x in range(cycles):
    model.compile(optimizer=tf.keras.optimizers.Adam(initialLR * (10**(-x)),
                  amsgrad=True),
                  loss=custom_loss,
                  metrics=[pt_error, eta_error, phi_error, e_error,dPT, dR])
    model.summary()
    
    model.fit(
        trainIn,
        trainOut,
        validation_data=(
            valIn,
            valOut),
        epochs=epochs,
        batch_size=64,
        callbacks=[callback,tensorboard_callback])

model.save(outputModelName)
