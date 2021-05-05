"""
This script extracts the information on the raw reco, reco, and gen jets from
data.npy and saves the raw reco jets in rawJets.npy, the gen jets in
finalJets.npy, and the reco jets in initialjets.npy.
"""

import numpy as np

def main():
    #Load data
    dataInitial = np.load("data.npy")
    finalJets = []
    initialJets = []
    rawJets = []
    #Iterate over the data
    for x in range(len(dataInitial)):
        #Test if a matched jet
        if(dataInitial[x,1] != 0 or dataInitial[x,2] != 0 or 
           dataInitial[x,3] != 0 or dataInitial[x,4] != 0 or 
           dataInitial[x,5] != 0 or dataInitial[x,0] != 0):
            #Test if start of a new jet
            if(dataInitial[x,0] == 1):
                finalJets.append(np.array([dataInitial[x,-4], 
                                           dataInitial[x,-3], 
                                           dataInitial[x,-2],
                                           dataInitial[x,-1]]))
                initialJets.append(np.array([dataInitial[x,-8],
                                             dataInitial[x,-7],
                                             dataInitial[x,-6],
                                             dataInitial[x,-5]]))
                rawJets.append(np.array([dataInitial[x,-12],
                                         dataInitial[x,-11],
                                         dataInitial[x,-10],
                                         dataInitial[x,-9]]))
    
    #Save the data
    finalJets=np.array(finalJets)
    np.save("finalJets.npy", finalJets)
    
    initialJets=np.array(initialJets)
    np.save("initialJets.npy", initialJets)
    
    rawJets=np.array(rawJets)
    np.save("rawJets.npy", rawJets)

if(__name__=="__main__"):
    main()