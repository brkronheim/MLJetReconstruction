# MLJetReconstruction - using machine learning to reconstruct jets for CMS


## Setting up
First, follow the instruction [here](http://opendata.cern.ch/docs/cms-guide-docker) to set up docker and install CMSSW. Make sure to get cmssw_10_6_8_patch1, not the version given in the tutorial.

Then enter the docker container and clone the repo and compile it. Make sure you are in the directory CMSSW_10_6_8_patch1/src before cloning. 
```
git clone https://github.com/brkronheim/MLJetReconstruction.git
scram b                 # compiles the code
cd MLJetReconstruction/JetAnalyzer
```

## Running the code

To run the code simply execute.
```
cmsRun python/ConfFile_cfg.py
```
As currently setup, the code will make predictions using the the jet reconstruction algorithm and generate a dataset with these predictions. In order to change this to produce a dataset with the initial training data. Replace JetAnalyzer.cc in plugins with AltJetAnalyzer.cc in misc (change the name to JetAnalyzer.cc). Also, change filelist.txt in JetAnalyer/python to contain
root://eospublic.cern.ch///eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/100000/08278E4E-E4EF-E611-8BD7-FA163E3ABA64.root
Then, recomplie the code and run it. It will generate a text file mlData.txt. This can be transformed into an appropriate file for training by cleanParticleData.py in ml. The file output can then be used in jetClassification.py and jetRegression.py to produce the classification and regression networks used in the C++ code. ExtractNetworks.ipynb contains the code necesary to extract the data necesary to run the networks in C++. rocPlots.ipynb contains code to view the quality of the regression predictions. 

Running the code in its default configuration with 
root://eospublic.cern.ch///eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/100000/0C0A7523-6AEF-E611-BFEE-002590494E64.root 
in filelist.txt will generate predictions using the ml algorithm to a text file mlData.txt. Convert this into a .npy file and then you can train a network to perform corrections on it using jetCorrection.py. Finally, viewPredictions.ipynb can be used to view some histograms showing the performance of the network.

Beyond this, extractHistos.cc can be used to generate some Root Histograms showing how the matching process between gen and reco particles works, and extractHistos2.cc will make some histograms showing the algorithms performance. They are executed by running the following commands in the docker instance:

```
root -b # Opens root
.x extractHistos.cc # Runs the c++ compiler built into root, will generate the histograms
```

View the presentations folder to see presentations on how this project has changed over time.
