import numpy as np
import tensorflow as tf
import math
keras = tf.keras
from quantileNetwork import QuantileNet, makeDataset

#Declare network
network = QuantileNet()

#Add normal keras layers
network.add(keras.layers.Dense(50, activation="relu"))
network.add(keras.layers.Dense(50, activation="relu"))
network.add(keras.layers.Dense(50, activation="relu"))
network.add(keras.layers.Dense(1))

#We want 100 training examples
samples=100
#training data, 2 dimensional examples, Gaussian noise with sd 0.2
xValsT = np.array([np.linspace(-3,3,samples)]).T
noiseT = np.random.normal(0, 0.2,size=xValsT.shape)
yValsT = np.sin(xValsT*math.pi*2)*xValsT-np.cos(xValsT*math.pi) + noiseT
yValsT2 = xValsT**2/3-3 + noiseT

#Turn the normal data into a dataset better for the quantile network.
xValsT, yValsT = makeDataset(xValsT, #input x
                             np.concatenate([yValsT2,yValsT], axis=1), #input y
                             1, # x dims
                             2, # y dims
                             samples) # examples

#Repeat for validation data
xValsV = np.array([np.linspace(-3+3/samples,3-3/samples,samples-1)]).T
noiseV = np.random.normal(0, 0.2,size=xValsV.shape)
yValsV = np.sin(xValsV*math.pi*2)*xValsV-np.cos(xValsV*math.pi) + noiseV
yValsV2 = xValsV**2/3-3 + noiseV
xValsV, yValsV = makeDataset(xValsV, np.concatenate([yValsV2, yValsV], axis=1), 1, 2, samples-1)

#Train for 100 epochs, restore best weights
epochs=100
callbacks=[tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)]

#Train with iteratively decreased learning rate
for x in [0,1,2]:
    network.compile(optimizer=tf.keras.optimizers.Adam(0.01 * (10**(-x)),
                  amsgrad=True),
                  loss=network.loss,
                 run_eagerly=True)
    
    history = network.fit(
        xValsT,
        yValsT,
        validation_data=(
            xValsV,
            yValsV),
        epochs=epochs,
        callbacks=callbacks,
        batch_size=32)

#Save the network
network.save("Test")
