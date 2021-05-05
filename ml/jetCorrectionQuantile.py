""" 
Training code for the jet quantile correction network. This network is used to 
correct the algoritm's final predictions
"""

import click
import os

import random as rn
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from quantileNetwork import QuantileNet, makeDataset

@click.command()
@click.option('--hidden', default=5, help='Number of hidden layers')
@click.option('--width', default=50, help='Width of the hidden layers')
@click.option('--alpha', default=0.2, help='Slope for leaky relu')
@click.option('--initialLR', default=0.01, help='initial learning rate')
@click.option('--batch', default=512, help='batch size')
@click.option('--cycles', default=4, help='Number of cylces to train for')
@click.option('--epochs', default=100, help='Number of epochs in a cylce')
@click.option('--patience', default=20, help='Number of epochs with no improvement before ending')
@click.option('--datain', default="predicted1.npy", help='Name of input data file')
@click.option('--dataout', default="finalJets.npy", help='Name of output data file')
@click.option('--networkName', default="jet_regression_lrelu_512batch_0.01LR_4cycles", help='Name of network')


def main(hidden, width, alpha, initiallr, batch, cycles, epochs, patience, datain, dataout, networkname):
  
    outputModelName = networkname
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(42)
    rn.seed(12345)
    tf.random.set_seed(3)

    outputDims=1 


    #Declare network
    network = QuantileNet()
    
    network.add(
        tf.keras.layers.Dense(
            width,
            kernel_initializer="glorot_uniform"
            ))
    network.add(tf.keras.layers.LeakyReLU(alpha=alpha))

    for n in range(hidden - 1):
        network.add(
            tf.keras.layers.Dense(
                width,
                kernel_initializer="glorot_uniform"))
        network.add(tf.keras.layers.LeakyReLU(alpha=alpha))

    network.add(
        tf.keras.layers.Dense(
            outputDims,
            kernel_initializer="glorot_uniform",
            activation=None))
    
    xValsPt = np.load(datain)[:,:,0].reshape((-1))
    xValsEta = np.load(datain)[:,:,1].reshape((-1))
    xValsPhi = np.load(datain)[:,:,2].reshape((-1))
    xValsE = np.load(datain)[:,:,3].reshape((-1))
    
    xVals = np.concatenate([[xValsPt], [xValsEta], [xValsPhi], [xValsE]], axis=0).T
    
    yValsPt = np.load(dataout)[:-1,0].repeat(100,axis=0)
    yValsPt = np.load(dataout)[:-1,0].repeat(100,axis=0)
    yValsPt=np.log(np.where(yValsPt<0.01, 0.01, yValsPt))
    yValsEta = np.load(dataout)[:-1,1].repeat(100,axis=0)
    yValsPhi = np.load(dataout)[:-1,2].repeat(100,axis=0)
    yValsE = np.load(dataout)[:-1,3].repeat(100,axis=0)
    yValsE=np.log(np.where(yValsE<1, 1, yValsE))
    yVals = np.concatenate([[yValsPt], [yValsEta], [yValsPhi], [yValsE]], axis=0).T
    
    trainIn, _, trainOut, _ = train_test_split(xVals,
                                                            yVals,
                                                            test_size=89/100,
                                                            random_state=42)
    trainIn = trainIn.T
    trainOut = trainOut.T
    
    for x in range(4):
        trainIn[x] = (trainIn[x]-np.mean(trainIn[x]))/np.std(trainIn[x])
        trainOut[x] = (trainOut[x]-np.mean(trainOut[x]))/np.std(trainOut[x])
    
    trainIn = trainIn.T
    trainOut = trainOut.T
        
    trainIn, valIn, trainOut, valOut = train_test_split(trainIn,
                                                            trainOut,
                                                            test_size=1/11,
                                                            random_state=42)
    
    #Turn the normal data into a dataset better for the quantile network.
    xValsT, yValsT = makeDataset(trainIn, #input x
                                 trainOut, #input y
                                 4, # x dims
                                 4, # y dims
                                 len(trainIn)) # examples
 
    xValsV, yValsV = makeDataset(valIn, #input x
                                 valOut, #input y
                                 4, # x dims
                                 4, # y dims
                                 len(valIn)) # examples
    
    #Train for 100 epochs, restore best weights
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, restore_best_weights=True
    )]
    
    #Train with iteratively decreased learning rate
    for x in range(cycles):
        network.compile(optimizer=tf.keras.optimizers.Adam(initiallr * (10**(-x)),
                      amsgrad=True),
                      loss=network.loss,
                     run_eagerly=True)
        
        network.fit(
            xValsT,
            yValsT,
            validation_data=(
                xValsV,
                yValsV),
            epochs=epochs,
            callbacks=callbacks,
            batch_size=batch,
            verbose=2)
    
    #Save the network
    network.save(outputModelName)
