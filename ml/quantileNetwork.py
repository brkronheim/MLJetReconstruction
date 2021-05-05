import numpy as np
import tensorflow as tf
keras = tf.keras

def makeDataset(inputVals,outputVals, inputDims, outputDims, samples):
    """
    A function for turning typical machine learning training sets into
    quantile one. This function turns a typical input, output dataset into
    a dataset of the following input form:
        
    input, one hot encoding of output dim to predict, output dims already 
    predicted, zero padding
    
    The output is simply one value for each input, corresponding to the
    predicted value.
    
    ----------
    inputVals : inputs to the network
    outputVals: outputs of the network
    inputDims: dimension of input data
    outputDims: dimension of output data
    samples: number of training examples
        
    Returns
    -------
    dSetIn: the new input dataset
    dSetOut: the new output dataset
    """
    
    
    dSetIn=[]
    dSetOut=[]
    for x in range(outputDims):
        temp = [inputVals]
        for y in range(x):
            temp.append(np.zeros((samples,1)))
        
        temp.append(np.ones((samples,1)))
        
        for y in range(x+1,outputDims):
            temp.append(np.zeros((samples,1)))
        
        for y in range(0, x):
            temp.append(outputVals[:,y:y+1])
        
        for y in range(x, outputDims-1):
            temp.append(np.zeros((samples,1)))
        
        dSetIn.append(np.concatenate(temp, axis=1))
        dSetOut.append(outputVals[:,x:x+1])
        
    dSetIn = np.concatenate(dSetIn, axis=0)
    dSetOut = np.concatenate(dSetOut, axis=0)
    return(dSetIn,dSetOut)

class QuantileNet(keras.Model):
    """
    An implementation of a quantile network. For intput data D and output dims
    x, y, z, ..., this network learns p(x), p(y|x), p(z|x,y), ... and can be
    used to sample from p(x, y, z, ...)   
    
    """
    def __init__(self, gradLossScale=100):
        """
        ----------
        gradLossScale: value to scale the graident loss by. The gradient loss
            comes from negative slopes from the quantile function
            
        Returns
        -------
        None
        
        
        """
        super(QuantileNet, self).__init__()
        self.gradLossScale=gradLossScale
        self.netLayers=[]
    
    def add(self, layer):
        """
        Add a layer to the network
        ----------
        layer: a layer to add to the network
            
        Returns
        -------
        None
        
        """
        self.netLayers.append(layer)
    
    @tf.function
    def innerCall(self, inputs):
        """
        This function accepts inputs to the network and creates output. It also
        randomly picks and assigns quantiles to predict.

        Parameters
        ----------
        inputs : The inputs to the network

        Returns
        -------
        gradLoss : The loss associated with negative quantile slopes
        output : The output of the quantile network

        """
        output = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        count = 0
        
        #Find number of examples
        for inputVal in tf.transpose(inputs):
            count+=1
        gradLoss=np.float32(0.0)
        
        #Randomly sample quantiles
        quantiles=tf.random.uniform(shape = [1,count], minval=0, maxval=1, dtype=tf.float32)*6-3
        
        #Full inputs
        inputs = tf.transpose(tf.concat([inputs, quantiles], axis=0))
        
        #Make predictions and get gradients
        out = None
        with tf.GradientTape() as g:
            g.watch(inputs)
            val = inputs
            for layer in self.netLayers:
                val = layer(val)
            
            out=val
        
        #Grad loss
        grads=g.gradient(out, inputs)[:,-1]
        gradLoss+=tf.reduce_sum(self.gradLossScale*tf.math.square(tf.where(grads<0,grads,0)))
        
        #Output contains quantiles and inputs for use in loss function
        output = tf.concat([out, tf.transpose(quantiles), inputs], axis=1)
        return(gradLoss, output)
    
    
    def call(self, inputs):
        """
        This function registers the gradLoss and calls the network outside of
        the tf.function

        Parameters
        ----------
        inputs : The inputs to the network

        Returns
        -------
        outputVal : The output of the network

        """
        
        inputs = tf.transpose(inputs)
        gradLoss, outputVal = self.innerCall(inputs)
        self.add_loss(gradLoss)
        return outputVal
    
    
    def loss(self, y_actual,y_pred):
        """
        This function calcualtes the quantile loss. Note that the quantiles 
        give to the network are actually between -3 and 3 to maximize the 
        range they cover. This must be undone by the loss function
       
        Parameters
        ----------
        y_actual : The true outputs
        y_pred : The predicted outputs, along with the quantiles

        Returns
        -------
        outputVal : The output of the network
        
        """
        val = y_actual-tf.expand_dims(y_pred[:,0],1)
        quants = (tf.expand_dims(y_pred[:,1],1)+3)/6
        val = tf.where(val<0.0, (-1+quants)*val, (quants)*val)
        val = tf.reduce_mean(val)
        return(val)
    
def callNet(quantileObject, quantileSamples, inputs,inputCount, inputDims, outputDims):
    """
    This function is used for making predictions from quantile nets.
    
    Parameters
    ----------
    quantileObject : The quantile network object
    quantileSamples : The number of samples to take for each input
    inputs : Input data
    inputCount : Number of examples
    inputDims : Dimension of input data
    outputDims : Dimension of output distribution

    Returns
    -------
    output : The output of the quantile net. It has shape
        (inputCount, quantileSamples, outputDims)
    
    """
    
    #Initial input, sampling the first dimension
    finalInputs=[]
    samplingLocs=[[1.0]]
    for x in range(1,outputDims):
        samplingLocs.append([0.0])
    
    samplingLocs=tf.cast(samplingLocs, dtype=tf.float32)
    samplingLocs = tf.repeat(samplingLocs,quantileSamples, axis=1)
    
    for inputVal in tf.transpose(inputs):
        #Random quantile value
        randomQuant=(tf.random.uniform([1,quantileSamples], minval=-3, maxval=3, dtype=tf.dtypes.float32))
        #Input repeated according to number of samples
        extendedInput = tf.repeat(tf.expand_dims(inputVal,axis=1), quantileSamples, axis=1)
        #No data for these values yet
        zeroInputs = 0*tf.random.normal([outputDims-1,quantileSamples],dtype=tf.float32)
        #Combine the input components
        finalInputs.append(tf.squeeze(tf.concat([extendedInput, samplingLocs,zeroInputs, randomQuant], axis=0)))
    
    #Combine all the inputs
    finalInputs = tf.transpose(tf.concat(finalInputs, axis=1))
    
    def predict(val):
        """
        Main predict loop. Accepts as input the input data, returns the output data,
        """
        #Input already setup
        output = predict_inner(val)
        
        #Setup while loop to make general predictions
        i = tf.constant(1)
        condition = lambda i, currentState, currentProb: tf.less(i, outputDims)
        
        i, val,output = tf.while_loop(condition, predict2, 
                                                     [i, val, 
                                                      output])
        location = (i-1)%outputDims
        
        finalOutput = tf.concat([val[:,inputDims+outputDims:inputDims+outputDims+location], output, val[:,inputDims+outputDims+location+1:-1]], axis=1)
        return(finalOutput)
       
    def predict2(x,val, output):
        """
        The main prediction driver. This reorganizes the input to the network
        as dimensions are samples. It accepts a counter variables to stop sampling
        when the required dimensions have been sampled.
        """
        location = x%outputDims
        oldLocation = (x-1)%outputDims
        vectors=[val[:,0:inputDims]]#Raw input,1
        for y in range(location):#Not sampling here
            vectors.append(val[:,0:1]*0)
        vectors.append(val[:,0:1]*0+1)#Sampling here
        for y in range(location+1,outputDims):#Not sampling here, another2
            vectors.append(val[:,0:1]*0)
        vectors.append(val[:,inputDims+outputDims:inputDims+outputDims+oldLocation])#Previous coordinates
        vectors.append(output) #Just sampled coordinate
        vectors.append(val[:,inputDims+outputDims+oldLocation+1:-1])#Previous coordinates
        vectors.append(tf.random.uniform(output.shape,-3,3,tf.float32))#New quantiles
        val = tf.concat(vectors, axis=1)
        
        output = predict_inner(val)
        return(tf.add(x, 1), val,output)
     
    @tf.function
    def predict_inner(val):
        """
        Run interior predictions within graph
        """
        for layer in quantileObject.netLayers:
            val = layer(val)
        return(val)
    
    output = predict(finalInputs)
    output = tf.reshape(output, (inputCount, quantileSamples, outputDims))
    return(output)


