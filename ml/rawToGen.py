""" 
Training code for the raw to gen network. This network is used to
go from a raw reco jet to a gen jet
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #supress tensorflow warning messages
warnings.filterwarnings("ignore", category=FutureWarning) 

import click

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from math import pi


@click.command()
@click.option('--hidden', default=5, help='Number of hidden layers')
@click.option('--width', default=50, help='Width of the hidden layers')
@click.option('--alpha', default=0.2, help='Slope for leaky relu')
@click.option('--initialLR', default=0.01, help='initial learning rate')
@click.option('--batch', default=512, help='batch size')
@click.option('--cycles', default=4, help='Number of cylces to train for')
@click.option('--epochs', default=100, help='Number of epochs in a cylce')
@click.option('--patience', default=20, help='Number of epochs with no improvement before ending')
@click.option('--dataName', default="mlDataPredictionsNewNorm.npy", help='Name of input data file')
@click.option('--networkName', default="raw_to_gen_base", help='Name of network')


def main(hidden, width, alpha, initiallr, batch, cycles, epochs, patience, dataname, networkname):
    
    data = np.load("mlDataPredictionsNewNorm.npy").T
    rawRecoData = data[4:8,:]
    genData = data[12:16]
    
    rawRecoData[0,:] = np.log(rawRecoData[0,:])
    rawRecoData[3,:] = np.log(rawRecoData[3,:])
    genData[0,:] = np.log(genData[0,:])
    genData[3,:] = np.log(genData[3,:])
    
    inputData=rawRecoData
    outputData=genData
    
    trainIn, testIn, trainOut, testOut = train_test_split(inputData.T,
                                                        outputData.T,
                                                        test_size=1/10,
                                                        random_state=42)
    
    
    for x in [0,1,3]:
        trainIn[:,x]=(trainIn[:,x]-np.mean(trainIn[:,x]))/(np.std(trainIn[:,x]))
        trainOut[:,x]=(trainOut[:,x]-np.mean(trainOut[:,x]))/(np.std(trainOut[:,x]))
    
    
    
    
    def loss(y_actual, y_pred):
        custom_loss = tf.where(tf.math.square(tf.math.subtract(y_actual[:,2], y_pred[:,1]))<tf.math.square(tf.math.subtract(y_actual[:,2], y_pred[:,2]+2*pi)),
                               tf.math.subtract(y_actual[:,2], y_pred[:,2]), 
                               tf.math.subtract(y_actual[:,2], y_pred[:,2]+2*pi))
        custom_loss = tf.where(tf.math.square(custom_loss)<tf.math.square(tf.math.subtract(y_actual[:,2], y_pred[:,2]-2*pi)),
                               custom_loss,
                               tf.math.subtract(y_actual[:,2], y_pred[:,2]-2*pi))
        custom_loss = tf.reduce_mean(tf.math.square(custom_loss))
        
        custom_loss += tf.reduce_mean(tf.square(y_actual[:,0]-y_pred[:,0]))
        custom_loss += tf.reduce_mean(tf.square(y_actual[:,1]-y_pred[:,1]))
        custom_loss += tf.reduce_mean(tf.square(y_actual[:,3]-y_pred[:,3]))
        
        return custom_loss
    
    def ptLoss(y_actual, y_pred):
        return(tf.reduce_mean(tf.square(y_actual[:,0]-y_pred[:,0])))
    
    def etaLoss(y_actual, y_pred):
        return(tf.reduce_mean(tf.square(y_actual[:,1]-y_pred[:,1])))
    
    def phiLoss(y_actual, y_pred):
        custom_loss = tf.where(tf.math.square(tf.math.subtract(y_actual[:,2], y_pred[:,1]))<tf.math.square(tf.math.subtract(y_actual[:,2], y_pred[:,2]+2*pi)),
                               tf.math.subtract(y_actual[:,2], y_pred[:,2]), 
                               tf.math.subtract(y_actual[:,2], y_pred[:,2]+2*pi))
        custom_loss = tf.where(tf.math.square(custom_loss)<tf.math.square(tf.math.subtract(y_actual[:,2], y_pred[:,2]-2*pi)),
                               custom_loss,
                               tf.math.subtract(y_actual[:,2], y_pred[:,2]-2*pi))
        custom_loss = tf.where(custom_loss<0.0, (tf.transpose(y_actual)[1]-1)*custom_loss, tf.transpose(y_actual)[1]*custom_loss)
        custom_loss = tf.reduce_mean(custom_loss)
        return(custom_loss)
    
    def eLoss(y_actual, y_pred):
        return(tf.reduce_mean(tf.square(y_actual[:,3]-y_pred[:,3])))
    
    
    
    
    trainIn, valIn, trainOut, valOut = train_test_split(trainIn,
                                                        trainOut,
                                                        test_size=1/9,
                                                        random_state=2)
    
    
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
        model.compile(optimizer=tf.keras.optimizers.Adam(initiallr * (10**(-x)),
                      amsgrad=True),
                      loss=loss,
                      metrics=[ptLoss, etaLoss, phiLoss, eLoss])
        
        model.fit(
            trainIn,
            trainOut,
            validation_data=(
                valIn,
                valOut),
            epochs=epochs,
            batch_size=batch,
            callbacks=[callback])
    model.save(networkname)