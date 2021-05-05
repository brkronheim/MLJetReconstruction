"""
Runs the probabilistic prediction algorithm on the data in data.npy, saves
the predictions in predicted1.npy, and plots the results of the pt predictions.
"""

import numba
import click
import time
import os

import random as rn
import numpy as np

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)

@click.command()
@click.option('--counts', default=100, help='Number of predictions per'+
              ' example, default is 100')
@click.option('--maxjets', default=3, help='Maximum number of jets to'+
              ' predict, default is 3')

def main(counts, maxjets):
    
    fullStartTime = time.time()
    depth = 5
    counts = 100
    maxJets = maxjets
    matrices=[]
    biases=[]
    finalJets = np.load("finalJets.npy")
    initialJets = np.load("initialJets.npy")
    
    #Extract all the neural network parameters
    for x in range(depth):
        matrices.append(np.float64(np.loadtxt("classification/weights_" +str(x)+".txt", delimiter=",").T))
        biases.append(np.float64(np.loadtxt("classification/biases_" +str(x)+".txt", delimiter=",").T))
    matricesb = np.float64(np.loadtxt("classification/weights_" +str(x+1)+".txt", delimiter=",").T)
    biasesb = np.float64(np.loadtxt("classification/biases_" +str(x+1)+".txt", delimiter=",").T)
    
    matricesc = np.float64(np.loadtxt("classification/weights_" +str(x+2)+".txt", delimiter=",").T)
    biasesc = np.float64(np.loadtxt("classification/biases_" +str(x+2)+".txt", delimiter=",").T)
    
    matricesb = np.expand_dims(matricesb,1)
    matricesc = np.expand_dims(matricesc,0)
    matricesc = np.expand_dims(matricesc,0)
    biasesb = np.expand_dims(biasesb,0)
    biasesc = np.expand_dims(biasesc,0)
    matrices=tuple(matrices)
    biases=tuple(biases)
    

    matrices2=[]
    biases2=[]
    for x in range(depth):
        matrices2.append(np.float64(np.loadtxt("regression/weights_" +str(x)+".txt", delimiter=",").T))
        biases2.append(np.float64(np.loadtxt("regression/biases_" +str(x)+".txt", delimiter=",").T))
    
    matrices2b = [np.float64(np.loadtxt("regression/weights_" +str(x+1)+".txt", delimiter=",").T)]
    biases2b = [np.float64(np.loadtxt("regression/biases_" +str(x+1)+".txt", delimiter=",").T)]
    
    
    matrices2b = np.expand_dims(matrices2b[-1],1)
    biases2b = np.expand_dims(biases2b[-1],0)
    
    matrices2=tuple(matrices2)
    biases2=tuple(biases2)
     
    #Extract the normalization information
    normInfo1=np.loadtxt("classification/normInfo1.txt", delimiter=",")
    normInfo2=np.loadtxt("regression/normInfo1.txt", delimiter=",")
    normInfo3=np.loadtxt("regression/normInfo2.txt", delimiter=",")
    
    
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

    dataInitial=np.load("data.npy")
    
    #Extract the input data
    data=[]
    currentX=0.0
    jetSize=[]
    for x in range(len(dataInitial)):
        
        temp=[]
        #Test if a matched jet
        if(True or dataInitial[x,1] !=0 or dataInitial[x,2]!=0 or dataInitial[x,3]!=0 or dataInitial[x,4]!=0 or dataInitial[x,5]!=0 or dataInitial[x,0]!=0):
            #Test if start of a new jet
            if(dataInitial[x,0]==1):
                jetSize.append(len(data)-currentX)
                currentX=len(data)
        
    
            particlePT = (dataInitial[x,6]**2 + dataInitial[x,7]**2)**0.5
            particlePhi = 0.0
            particleEta = 0.0
            if(particlePT != 0):
                particlePhi = np.arcsin(dataInitial[x,7]/particlePT)
                particleEta = np.arcsinh(dataInitial[x,8]/particlePT)
            temp.append(np.log(particlePT))
            temp.append(particleEta)
            temp.append(particlePhi)
            temp.append(np.log(dataInitial[x,9]))
            temp.append(np.cbrt(dataInitial[x,10]))
            temp.append(np.cbrt(dataInitial[x,11]))
            temp.append(np.cbrt(dataInitial[x,12]))
            temp= temp + newPDGID[dataInitial[x,13]]
            
            data.append(np.array(temp))
        
    data=np.array(data).T
    
    #Put the particles for each test case in a list
    particles=[]
    particleCount=[]
    index = 0
    totalJets = len(jetSize)-1
    for x in range(totalJets):
        particles.append(data[:,index:index+jetSize[x+1]].T)
        particleCount.append(jetSize[x+1])
        index+=jetSize[x+1]
    particles = tuple(particles)
    particleCount = tuple(particleCount)
    
    
    @numba.njit()
    def regressPredict(val):
        """
        Makes a prediction from the regression network

        Parameters
        ----------
        val : array of floats
            Input to the regression network.
 
        Returns
        -------
        val : array of floats
            Output of the regression network

        """
       
        
        
        x=0
        while(x<depth):
            val = np.dot(val, matrices2[x]) #Multiply
            val = np.add(val, biases2[x]) #Add
            val = np.where(val<0, val*0.2, val) #Activation
            x+=1
        
        val = np.dot(val, matrices2b)
        val = np.add(val, biases2b)
        val = np.expand_dims(val, 0)
        
        return(val)
    
    @numba.njit()
    def classPredict(val):
        """
        Makes a prediction from the classifcation network

        Parameters
        ----------
        val : array of floats
            Input to the classifcation network.
 
        Returns
        -------
        val : array of floats
            Output of the classifcation network

        """
        x=0
        while(x<depth):
            val = np.dot(val, matrices[x])
            val = np.add(val, biases[x])
            val = np.where(val<0, val*0.2, val)
            x+=1
        val = np.dot(val, matricesb)
        val = np.add(val, biasesb)
        val = np.dot(val, matricesc)
        val = np.add(val, biasesc)
        val = 1 / (1 + np.exp(-val))
        return(val)
    
    
    @numba.njit()
    def quantRegressPredict(val, count):
        """
        Makes a prediction from the quantile regression network

        Parameters
        ----------
        val : array of floats
            Input to the classifcation network.
        count : int
            Number of predictions
 
        Returns
        -------
        outVal : array of floats
            Output of the quantile regression network

        """
        
        #Normalize input data
        for x in range(29):
            val[:,x] = (val[:,x]-normInfo2[x,0])/normInfo2[x,1]
        
        indicator = np.zeros((count, 7)) #Indicates which prediction to make
        indicator[:,0] = np.ones((count))
        
        quant = np.random.uniform(-3,3,(count)) #Random quantile
        
        #New input
        newVal = np.zeros((count, 37))
        newVal[:,:29] = val
        newVal[:,29:36] = indicator
        newVal[:,36] = quant
        
        #Will hold the output
        outVal = np.zeros((count, 4))
        
        x=0
        while(x<3):
            #Make the quantile predictions
            val = regressPredict(newVal.copy())
    
            newVal[:,29+x] = np.zeros((count))
            newVal[:,29+x+1] = np.ones((count))
            newVal[:,33+x] = val[:,0]
            outVal[:,x] = val[:,0]
            quant = np.random.uniform(-3,3,(count))
            newVal[:,36] = quant
            x+=1
            
        #Final prediction   
        val = regressPredict(newVal, count)
        outVal[:,x] = val[:,0]
        
        #Unnormalize prediction
        for x in range(4):
            outVal[:,x] = outVal[:,x]*normInfo3[x,1] + normInfo3[x,0]
        
        return(outVal)
    
    @numba.njit()
    def quantClassPredict(val, count):
        """
        Makes a prediction from the quantile classification network

        Parameters
        ----------
        val : array of floats
            Input to the classifcation network.
        count : int
            Number of predictions
 
        Returns
        -------
        outVal : array of floats
            Output of the quantile classification network

        """
        #Normalize
        for x in range(29):
            val[:,x] = (val[:,x]-normInfo1[x,0])/normInfo1[x,1]
        
        #Setup input
        indicator = np.zeros((count, 1))
        indicator[:,0] = np.ones((count))
        quant = np.random.uniform(-3,3,(count))
        newVal = np.zeros((count, 31))
        newVal[:,:29] = val
        newVal[:,29:30] = indicator
        newVal[:,30] = quant
        x=0
        outVal = np.zeros((count, 1))
        
        #make predition
        val = classPredict(newVal)
        outVal[:,x] = val[:,0]
        
        #Round output
        outVal = np.where(outVal<0.4, 0*outVal, outVal*0+1)
        return(outVal)
    
    
    @numba.njit()
    def alg(particleList, particleCount, sampleCount, maxJets):
        """
        Runs the full prediction code

        Parameters
        ----------
        particleList : list of arrays
            List of arrays of the input particles
        particleCount : int
            Number of input particles
        sampleCount : int
            Number of samples
        maxJets : int
            Maximum number of jets to predict
 
        Returns
        -------
        outputs : array of floats
            Predicted jets
        jetCounts : array of ints
            Number of predicted jets for each sample

        """
        currentSample = 0 # Start with the first sample
        jetCounts = np.ones((sampleCount)) # Store how many jets each sample has
        outputs = np.zeros((maxJets,sampleCount,4)) # Store the current output jets
        for currentSample in range(sampleCount):
            currentSeed = np.zeros((maxJets,7)) #Seed of current sample
            currentJetCount = 0 #No jets yet
            currentParticle = 0 #Start with the first particle
            inputVector = np.zeros((1,29)) # Will serve as the input to the networks
            while(currentParticle < particleCount): # Go over each particle
                inputVector[0,11:] = particleList[currentParticle] # Put the current particle in the input vector
                particleAdded = False # Particle has not been added yet
                currentJet = 0 #Start with the first jet
                while(not particleAdded and currentJet < maxJets): #Check if we've added the particle or reached the max number of jets
                    inputVector[0,0:7] = currentSeed[currentJet,:] #Set the seed particle for the current jet
                    inputVector[0,7:11] = outputs[currentJet, currentSample,:] #Set the current jet
                    
                    if(currentParticle == 0): #If this is the first particle it goes in the first jet
                        particleAdded = True #Particle will be added
                        currentSeed[0,:] = particleList[currentParticle,0:7].copy() #The jet seed is from the current particle
                        inputVector[0,0:7] = particleList[currentParticle,0:7].copy() #Set the seed for the first jet
                        inputVector[0,7:11] = [np.log(0.01),0,0,np.log(0.01)] #Set the jet to start at 0
                        currentJetCount += 1 #Added a jet
                        jetCounts[currentSample] = 1 #We have one jet
                    elif (currentJet == jetCounts[currentSample] ):
                        jetCounts[currentSample]+=1
                        particleAdded = True #Particle will be added
                        currentSeed[currentJet,:] = particleList[currentParticle,0:7].copy() #The jet seed is from the current particle
                        inputVector[0,0:7] = particleList[currentParticle,0:7].copy() #Set the seed for the first jet
                        inputVector[0,7:11] = [np.log(0.01),0,0,np.log(0.01)] #Set the jet to start at 0
                        currentJetCount += 1 #Added a jet
                    elif(quantClassPredict(inputVector.copy(), 1)[0] == 1): #Use the classifier to detrmine if we should add the jet
                        particleAdded = True
                    else:
                        particleAdded = False
                    
                    if(particleAdded): #Get the new jet
                        newJet = quantRegressPredict(inputVector.copy(), 1)
                        outputs[currentJet, currentSample,:] = newJet #Save the new jet
                    currentJet +=1
                
                
                currentParticle += 1
            
        return(outputs, jetCounts)
    
    allPredicted=[]
    allTrue=[]
    allInit=[]
    #Run the algorithm
    for v in range(totalJets):
        if(v%int(totalJets/1000)==0):
            print(100*v/totalJets)
            #Print percent complete every 0.1%
        
        #Make and store predictions
        out, jetCounts = alg(particles[v], particleCount[v], counts,maxJets)
        allTrue.append(finalJets[v+1,0])
        allInit.append(initialJets[v+1,0])
        allPredicted.append(out[0,:,:])

    #Make predictions
    allTrue=np.array(allTrue)
    allInit=np.array(allInit)
    allPredicted1=np.array(allPredicted)
    np.save("predicted1.npy", allPredicted1)
    #Plot distribution of predicted pt versus true and reco
    plt.figure()
    plt.hist(allTrue, range=(0,1200), bins=50, histtype="step", label="true")
    plt.hist(allInit, range=(0,1200), bins=50, histtype="step", label="reco")
    plt.hist(allPredicted1, range=(0,1200), weights = allPredicted1*0+1/counts, bins=50, histtype="step", label="predicted")
    plt.legend()
    plt.show()
    print("Time:", time.time()-fullStartTime)

  
if(__name__=="__main__"):
    main()