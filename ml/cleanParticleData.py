import numpy as np
import matplotlib.pyplot as plt

fileName = "mlDataNoDups"

data = np.loadtxt(fileName + ".txt")
np.save(fileName + "npy", data)

dataInitial=np.load(fileName + ".npy")
data=[]

temp=[]

for x in range(len(dataInitial)): # Iterate over all the data
    if(dataInitial[x,0]==1): # Check if this is the start of a new jet  
        sortedArray=np.argsort(np.array(val)) #Sort the old jet by pT
        real=False
        for y in range(len(val)):
            newVal=temp[sortedArray[len(val)-1-y]]
            newVal[0]=0
            if(not real): # Find first reco particle with a matching fen particle, otherwise remove
                if(newVal[1]!=0.0 or newVal[2]!=0.0 or newVal[3]!=0.0 or newVal[4]!=0.0 or newVal[5]!=0.0):
                    real=True
                    newVal[0]=1
                    if(len(data)<1000):
                        print("new", len(data))#, newVal) 
            if(real):
                data.append(newVal)
                
dataInitial=np.array(data)   

np.save(fileName + "Clean.npy", dataInitial)
