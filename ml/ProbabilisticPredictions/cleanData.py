"""
This script extracts the matching info from mlData.txt and saves the particles
sorted by pT for each event, starting with the first reco particle with a match
"""

import numpy as np

def main():
    #Open text file, save a .npy file for faster future loading
    dataInitial=np.loadtxt("mlData.txt")
    np.save("mlData.npy", dataInitial)
    
    data=[]
    temp=[]
    val = []
    
    for x in range(len(dataInitial)):
        if(dataInitial[x,0] == 1): #Start of a jet
            #Obtain sorted order of reco particles
            sortedArray = np.argsort(np.array(val))
            real = False
            for y in range(len(val)):
                #Start adding reco particles and matches starting with the
                #first reco particle with a match
                newVal = temp[sortedArray[len(val) - 1 - y]]
                newVal[0] = 0
                if(not real):
                    if(newVal[1] != 0.0 or newVal[2] != 0.0 or newVal[3] != 0.0
                       or newVal[4] != 0.0 or newVal[5] != 0.0):
                        real = True
                        newVal[0] = 1
                    
                if(real):
                    data.append(newVal)
            temp=[]
            val=[]
        temp.append(dataInitial[x,:])
        val.append((dataInitial[x,6]**2 + dataInitial[x,7]**2)**0.5)
    dataInitial=np.array(data)   
    #Save data as data.npy
    np.save("data.npy", dataInitial)
    
if(__name__=="__main__"):
    main()