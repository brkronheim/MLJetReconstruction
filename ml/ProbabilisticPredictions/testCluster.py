"""
Given some sample output data from the algorithm, this script applies the
clustering algorithm on it and generates some plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


font = {'family' : 'serif',
        'size'   : 12}

matplotlib.rc('font', **font)
data = np.load("sampleClusterData.npy")

out = data
jetCounts=3


#Data for clustering
pt = data[:,:,0].reshape([-1])
eta = data[:,:,1].reshape([-1])
phi = data[:,:,2].reshape([-1])
e = data[:,:,3].reshape([-1])

def clusterAlg(allJets, centers):
    """
    Accepts allJets, a list of all the jets output from the algorithm and 
    centers, a first guess as to the center. 
    
    Returns clusters, a list of lists of the jets in each cluster
    
    """
    
    #Calculate means and standard deviations for normalization
    mean1=np.mean(allJets[:,0])
    mean2=np.mean(allJets[:,1])
    mean3=np.mean(allJets[:,2])
    sd1=np.std(allJets[:,0])
    sd2=np.std(allJets[:,1])
    sd3=np.std(allJets[:,2])
    
    #Normalize pt, eta and phi
    newCenters = []
    for val in centers:
        val = np.array([(val[0]-mean1)/sd1,(val[1]-mean2)/sd2,(val[2]-mean3)/sd3])
        newCenters.append(val)
    centers = np.array(newCenters)
    allJets[:,0] = (allJets[:,0]-mean1)/sd1
    allJets[:,1] = (allJets[:,1]-mean2)/sd2
    allJets[:,2] = (allJets[:,2]-mean3)/sd3
    
    clusters=[]
    for x in range(len(centers)):
        clusters.append([])
    
    #Sort jets into clusters based on proxmity to starting centers
    for x in range(len(allJets[:,0])):
        collection = 0
        distance = np.sum((allJets[x,1:3]-centers[0,1:3])**2)**0.5
        for y in range(1,len(centers)):
            newDist = np.sum((allJets[x,1:3]-centers[y,1:3])**2)**0.5
            
            
            if(newDist<distance):
                distance = newDist
                collection = y
        clusters[collection].append(allJets[x,:])
    
    #Run for 10 iterations
    for z in range(10):
        collapses=0
        for x in range(len(clusters)):
            clusters[x] = np.array(clusters[x])
        clusters2=[]
        for x in range(len(clusters)):
            if(len(clusters[x].shape)>1):
                clusters2.append(clusters[x])
            else: #Cluster with nothing in it
                collapses+=1
        
        #Calculate new centers and the standard deviations along their axis
        centers=[]
        sds=[]
        clusters=[]
        newCluster=[]
        for x in range(len(clusters2)):
            temp = []
            temp2 = []
            for y in range(4):
                temp.append(np.mean(clusters2[x][:,y]))
                temp2.append(np.std(clusters2[x][:,y]))
            centers.append(np.array(temp))
            sds.append(np.array(temp2))
            clusters.append([])
        centers = np.array(centers)
        sds = np.array(sds)
        #Assign jets to new clusters based on distance relative toe standard
        #deviation
        for x in range(len(allJets[:,0])):
            collection = 0
            distance = np.sum(((allJets[x,:3]-centers[0,:3])/sds[0,:3])**2)
            for y in range(1,len(clusters2)):
                newDist = np.sum(((allJets[x,:3]-centers[y,:3])/sds[0,:3])**2)
                
                if(newDist<distance):
                    distance = newDist
                    collection = y
            clusters[collection].append(allJets[x,:])
        #Assigned collapsed clusters an array with just one 0
        for x in range(collapses):
            clusters.append(np.array([0]))
        clusters.append(newCluster)
    #Unnormalize the data
    for x in range(len(clusters)):
        newVal=[]
        current = clusters[x]
        if(len(np.array(current).shape)>1):
            for val in current:
                temp = [val[0]*sd1+mean1,val[1]*sd2+mean2,val[2]*sd3+mean3,val[3]]
                newVal.append(np.array(temp))
        clusters[x] = newVal
    return(clusters)

#Perform a 3d binning and locate the bins with more counts than all their
#boardering bins
bins = 10
rejected = 0
accepted = 0
maxima=[]
counts=[]
vals, [ptEdge, etaEdge, phiEdge] = np.histogramdd(np.array([pt,eta,phi]).T, bins=bins)
for x in range(0,bins):
    for y in range(0, bins):
        for z in range(0,bins):
            val = vals[x,y,z]
            add = True
            aVals = [0]
            bVals = [0]
            cVals = [0]
            if(x!=0):
                aVals.append(-1)
            if(y!=0):
                bVals.append(-1)
            if(z!=0):
                cVals.append(-1)
            if(x!=bins-1):
                aVals.append(1)
            if(y!=bins-1):
                bVals.append(1)
            if(z!=bins-1):
                cVals.append(1)
            
                
            for a in aVals:
                for b in bVals:
                    for c in cVals:
                        if(val<vals[x+a, y+b, z+c] or val == 0):
                            add = False
                            
                            
            if(add and val>1):
                accepted+=1
                maxima.append([(ptEdge[x]+ptEdge[x+1])/2, (etaEdge[y]+etaEdge[y+1])/2, (phiEdge[z]+phiEdge[z+1])/2])
                counts.append(val)
            else:
                rejected+=1

#Remove bins too close to other bins
maxima = [m for _,m in sorted(zip(counts,maxima), reverse=True)]
maxima = np.array(maxima)
centers=[]
for x in range(len(maxima)):
    add = True
    for val in centers:
        if((np.abs(maxima[x,1]-val[1])**2 and np.abs(maxima[x,2]-val[2])**2)**0.5<0.4 and (np.abs(np.exp(maxima[x,0])/np.exp(val[0]))<5/4 and np.abs(np.exp(maxima[x,0])/np.exp(val[0]))>4/5)):
            add = False
    if(add):
        centers.append(maxima[x])
maxima=np.array(centers).T
print("Accepted:", accepted)
print("Rejected:", rejected)



#Run the clustering algorithm
clusters = clusterAlg(np.array([pt, eta, phi, e]).T,maxima.T)

#Make plots of the original data, the binning results, and the clustering
#results
plt.figure()
for y in range(int(np.amax(jetCounts))):
    plt.scatter(out[y,:,1], out[y,:,2])

plt.xlabel("eta")
plt.ylabel("phi")
plt.tight_layout()
plt.savefig("eta_phi_alg.png")
plt.show()




plt.figure()
plt.hist2d(eta, phi, bins=bins)
plt.scatter(maxima[1,:], maxima[2,:], color="k",marker="x" )
plt.xlabel("eta")
plt.ylabel("phi")
plt.tight_layout()
plt.savefig("eta_phi_hist.png")
plt.show()


plt.figure()
for y in range(len(clusters)):
    cluster1 = np.array(clusters[y])
    if(len(cluster1.shape)>1):
        plt.scatter(cluster1[:,1],cluster1[:,2])#, s=1 )
plt.scatter(maxima[1,:], maxima[2,:], color="k",marker="x" )
plt.xlabel("eta")
plt.ylabel("phi")
plt.tight_layout()
plt.savefig("eta_phi_cluster.png")
plt.show()

plt.figure()
for y in range(int(np.amax(jetCounts))):
    plt.scatter(out[y,:,1], out[y,:,0])

plt.xlabel("eta")
plt.ylabel("log(pt)")
plt.tight_layout()
plt.savefig("eta_pt_agl.png")
plt.show()



plt.figure()
plt.hist2d(eta, pt, bins=bins)
plt.scatter(maxima[1,:], maxima[0,:], color="k",marker="x" )
plt.xlabel("eta")
plt.ylabel("log(pt)")
plt.tight_layout()
plt.savefig("eta_pt_hist.png")
plt.show()

plt.figure()
for y in range(len(clusters)):
    cluster1 = np.array(clusters[y])
    if(len(cluster1.shape)>1):
        plt.scatter(cluster1[:,1],cluster1[:,0])#, s=1 )
plt.scatter(maxima[1,:], maxima[0,:], color="k",marker="x" )
plt.xlabel("eta")
plt.ylabel("log(pt)")
plt.tight_layout()
plt.savefig("eta_pt_cluster.png")
plt.show()


plt.figure()
for y in range(int(np.amax(jetCounts))):
    plt.scatter(out[y,:,2], out[y,:,0])


plt.xlabel("phi")
plt.ylabel("log(pt)")
plt.tight_layout()
plt.savefig("phi_pt_alg.png")
plt.show()

plt.figure()
plt.hist2d(phi, pt, bins=bins)
plt.scatter(maxima[2,:], maxima[0,:], color="k",marker="x" )

plt.xlabel("phi")
plt.ylabel("log(pt)")
plt.tight_layout()
plt.savefig("phi_pt_hist.png")
plt.show()

plt.figure()
for y in range(len(clusters)):
    cluster1 = np.array(clusters[y])
    if(len(cluster1.shape)>1):
        plt.scatter(cluster1[:,2],cluster1[:,0])#, s=1 )
plt.scatter(maxima[2,:], maxima[0,:], color="k",marker="x" )
plt.xlabel("phi")
plt.ylabel("log(pt)")
plt.tight_layout()
plt.savefig("phi_pt_cluster.png")
plt.show()