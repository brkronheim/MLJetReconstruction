import numpy as np
import tensorflow as tf

import math

import matplotlib.pyplot as plt

from quantileNetwork import QuantileNet, callNet
keras = tf.keras
    

modelName = "Test" #Network name
model = QuantileNet() #Used to load network
counts=1000 #Test spots
testIn=np.array([np.linspace(-3,3,counts)]) #Test input, does not need modified for quantile net

testIn=tf.cast(testIn, tf.float32)

#Load model
newModel = tf.keras.models.load_model(modelName, custom_objects={"QuantileNet": QuantileNet, "loss": model.loss})

#1000 samples per input
sampleNum=1000
#Call the network
out=callNet(newModel, #Network
            sampleNum, #Number of samples
            tf.cast(testIn, tf.float32), #Input
            testIn.shape[1], #Number of examples (batch size)
            testIn.shape[0], #1d input
            1) # 2d output

output=out
out = tf.squeeze(out[:,:,0]) #Look at output
means = tf.reduce_mean(out, axis=1) #Means

meansSquared = tf.reduce_mean(out**2, axis=1)
sd = np.sqrt(np.array(meansSquared-means**2)) #Sd
outVals = np.reshape(np.array(out),[-1]) #All outputs

inVals = np.repeat(testIn, sampleNum) #All inputs


#Scatter plot of all output, along with mean, and the true mean
plt.figure()
plt.scatter(inVals, outVals, s=0.01, label="predicted scatter")
plt.plot(inVals, np.sin(inVals*math.pi*2)*inVals-np.cos(inVals*math.pi), color="k", label="true")
plt.plot(np.squeeze(testIn), np.squeeze(means), color="r", label="predicted mean")
plt.legend()
plt.show()

plt.figure()
plt.hist2d(inVals, outVals, bins=100, label="predicted scatter")
plt.plot(inVals, np.sin(inVals*math.pi*2)*inVals-np.cos(inVals*math.pi), color="k", label="true")
plt.plot(np.squeeze(testIn), np.squeeze(means), color="r", label="predicted mean")
plt.legend()
plt.show()

#Plot of true sd and predicted sd
plt.figure()
plt.plot(np.squeeze(testIn), sd, label="preidcted sd", color="r")
plt.plot(np.squeeze(testIn), np.abs(np.squeeze(testIn)*0+0.1), label="true sd", color="k")
plt.legend()
plt.show()
