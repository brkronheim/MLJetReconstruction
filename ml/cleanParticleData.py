""" 
Converts a text file output by the C++ code into a .npy file where the particles in a jet are sorted by pT
"""

import click
import numpy as np
import matplotlib.pyplot as plt

@click.command()
@click.option('--filename', default="gitParticleData", help='name of file to fix')

def main(filename):
    dataInitial=np.load(filename + ".npy")
    data=[]

    temp=[]
    val = []
    
    for x in range(len(dataInitial)):
        if(dataInitial[x,0]==1):
            sortedArray=np.argsort(np.array(val))
            real=False
            for y in range(len(val)):
                newVal=temp[sortedArray[len(val)-1-y]]
                newVal[0]=0
                if(not real):
                    if(newVal[1]!=0.0 or newVal[2]!=0.0 or newVal[3]!=0.0 or newVal[4]!=0.0 or newVal[5]!=0.0):
                        real=True
                        newVal[0]=1
                    
                if(real):
                    data.append(newVal)
            temp=[]
            val=[]
        temp.append(dataInitial[x,:])
        val.append((dataInitial[x,6]**2+dataInitial[x,7]**2)**0.5)
    dataInitial=np.array(data)   

    np.save(filename + "Clean.npy", dataInitial)
    
if(__name__=="__main__"):
    main()
