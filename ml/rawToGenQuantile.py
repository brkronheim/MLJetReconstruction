""" 
Training code for the raw to gen quantile network. This network is used to
go from a raw reco jet to a gen jet
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #supress tensorflow warning messages
warnings.filterwarnings("ignore", category=FutureWarning) 

import click

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from quantileNetwork import QuantileNet

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
@click.option('--networkName', default="raw_to_gen_full_v2_lrelu_5096batch_0.01LR_4cycles", help='Name of network')


def main(hidden, width, alpha, initiallr, batch, cycles, epochs, patience, dataname, networkname):



    data = np.load(dataname).T
    rawRecoData = data[4:8,:]
    genData = data[12:16]
    
    rawRecoData[0,:] = np.log(rawRecoData[0,:])
    rawRecoData[3,:] = np.log(rawRecoData[3,:])
    genData[0,:] = np.log(genData[0,:])
    genData[3,:] = np.log(genData[3,:])
    
    inputData=rawRecoData
    outputData=genData
    
    trainIn = inputData.T
    trainOut = outputData.T
    
    trainIn, testIn, trainOut, testOut = train_test_split(trainIn,
                                                        trainOut,
                                                        test_size=1/9,
                                                        random_state=42)
    normInfoOut=[[0,1],[0,1],[0,1],[0,1]]
    weights=[[],[],[],[]]
    normInfoOut=[[0,1],[0,1],[0,1],[0,1]]
    
    for x in [0,3]:
        weights[x]=np.exp(trainOut[:,x])
    for x in [1,2]:
        weights[x] = 1+0*trainOut[:,x]
    weights[0]=weights[0]/np.mean(weights[0])
    weights[3]=weights[3]/np.mean(weights[3])
    x = [trainOut[:,0]]
    
    for x in [0,1,3]:
        trainIn[:,x]=(trainIn[:,x]-np.mean(trainIn[:,x]))/(np.std(trainIn[:,x]))
        normInfoOut[x] = [np.mean(trainOut[:,x]), np.std(trainOut[:,x])]
        trainOut[:,x]=(trainOut[:,x]-np.mean(trainOut[:,x]))/(np.std(trainOut[:,x]))
    weights[0]=weights[0]/np.mean(weights[0])
    weights[3]=weights[3]/np.mean(weights[3])
    x = [trainOut[:,0]]
    y = [trainOut[:,1]]
    z = [trainOut[:,2]]
    w = [trainOut[:,3]]
    
    
    inVal = np.zeros(shape=np.array(x).shape)
    dataSetW = np.concatenate([trainIn.T, inVal+1, inVal, inVal, inVal, inVal, inVal, inVal, x], axis=0)
    dataSetZ = np.concatenate([trainIn.T, inVal, inVal+1, inVal, inVal, x, inVal, inVal, y], axis=0)
    dataSetY = np.concatenate([trainIn.T, inVal, inVal, inVal+1, inVal, x, y, inVal, z], axis=0)
    dataSetX = np.concatenate([trainIn.T, inVal, inVal, inVal, inVal+1, x, y, z, w], axis=0)
    dataSet = np.concatenate([dataSetW, dataSetZ, dataSetY, dataSetX], axis=1)
    
    
    trainIn = dataSet[:-1,:].T
    trainOut = np.expand_dims(dataSet[-1,:],1)
    
    
    trainIn, valIn, trainOut, valOut = train_test_split(trainIn,
                                                        trainOut,
                                                        test_size=1/10,
                                                        random_state=42)
    
    
    model = QuantileNet()
    
    model.add(
        tf.keras.layers.Dense(
            width,
            kernel_initializer="glorot_uniform"
            ))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))

    for n in range(hidden - 1):
        model.add(
            tf.keras.layers.Dense(
                width,
                kernel_initializer="glorot_uniform"))
        model.add(tf.keras.layers.LeakyReLU(alpha=alpha))

    model.add(
        tf.keras.layers.Dense(
            1,
            kernel_initializer="glorot_uniform",
            activation=None))
    
    callbackMetric="val_loss"
    callback = tf.keras.callbacks.EarlyStopping(
            monitor=callbackMetric, patience=patience, restore_best_weights=True)
    trainOut = tf.expand_dims(trainOut,1)
    valOut = tf.expand_dims(valOut,1)
    
    #Train for the desired number of cycles
    for x in range(cycles):
        model.compile(optimizer=tf.keras.optimizers.Adam(initiallr * (10**(-x)),
                      amsgrad=True),
                      loss=model.loss,
                     run_eagerly=False)
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
    