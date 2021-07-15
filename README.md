# MLJetReconstruction - using machine learning to reconstruct jets for CMS


## Setting up
First, follow the instruction [here](http://opendata.cern.ch/docs/cms-guide-docker) to set up docker and install CMSSW. Make sure to get cmssw_10_6_8_patch1, not the version given in the tutorial.

Then enter the docker container and clone the repo and compile it. Make sure you are in the directory CMSSW_10_6_8_patch1/src before cloning. 
```
git clone https://github.com/brkronheim/MLJetReconstruction.git
scram b                 # compiles the code
cd MLJetReconstruction/JetAnalyzer
```

## Running the Mean Buildup Model

The first code that should be run is the generation of the gen-reco particle matching dataset. This is done through the c++ code and can be executed as
```
cmsRun python/ConfFile_cfg.py executionMode=0
```
This tells the c++ code to use the first and smaller dataset to generate the gen-reco matches. It will save the data to mlData.txt. This should be transfered to the Python work area. Once there, run cleanParticleData.py as 
```
python cleanParticleData.py --filename=filename
```
where filename is whatever the mlData.txt file was saved as after the transfer. Do not include the .txt ending. The data in the file will be reordered so that within each jet the particles are orderd by descending gen pt. Additionally, it will be saved in a .npy file.

After this, the first two networks can be trained. Do this using the code
```
python jetClassification.py
python jetRegression.py
```
Note that both of these files have command line options which can be accessed with the --help flag. The defaults though are the values used currently.

Once these networks have been trained they need to be extracted using extracteNetworks.py. Run
```
python extractNetworks.py --filename=filename --regression=regression --classification=classification
```
Here filename is the name of the data file used to train the two networks, regression is the name of the regression network, and classification is the name of the classifcation network. This will store the weights and biases for the networks in a folder called classification and one called regression. The data contained therein should be transfered to the similarly names folders in JetAnalyzer/data. The data currently there is what you get from training with the default options.

You can make next some plots showing how the classifier performs in rocPlots.ipynb.

Once this has been done, run the c++ code again to actually make predictions using the ml algorithm. It will use the regression and classification networks previously trained to determine how to assemble the jet. Run the code by doing
```
cmsRun python/ConfFile_cfg.py executionMode=1
```
This will use a second, larger data file as its source. 

Once done with this, transfer the mlData.txt file as done previously. This file needs to be converted to a .npy file. The easiest way to do this is to execute the first three cells in performancePlots.ipynb, with the optional code uncommented. 

After this, train the last neural network by executing

```
python jetCorrection.py
```
Again, other options are availabe through --help

After this, you can either run the code in performancePlots.ipynb with the networks trained and the data already obtained, or you can run the c++ code in executionMode=2 to obtain a test set. This code will make a plot of performance plots.

Beyond this, extractHistos.cc can be used to generate some Root Histograms showing how the matching process between gen and reco particles works, and extractHistos2.cc will make some histograms showing the algorithms performance. They are executed by running the following commands in the docker instance:

```
root -b # Opens root
.x extractHistos.cc # Runs the c++ compiler built into root, will generate the histograms
```

Some additional plots showing the performance of this model can be seen in basicAlgRawToGenPredictionsGit.ipynb. When running it, make sure you use the name of the model you saved.

## Running the Mean Direct Model
Use the .npy file created for training the correction network. To train the network run
```
python rawToGen.py
```

Check the click arguments and make sure you are giving it the correct file name. Once trained, you can check its performance with basicRawToGenPredictionsGit.ipynb

## Running the Distribution Direct Model
Use the .npy file created for training the correction network. To train the network run
```
python rawToGenQuantile.py
```

Check the click arguments and make sure you are giving it the correct file name. Once trained, you can check its performance with quantileRawToGenPredictionsGit.ipynb

## Running the Distribution Buildup Model
Use the files created for training the first sets of regression and classification networks. To train the networks run
```
python jetClassificationQuantile.py
python jetRegressionQuantile.py
```
In order to make a good test set run
```
cmsRun python/ConfFile_cfg.py executionMode=2
```
This tells the c++ code to use the larger dataset to generate gen-reco matches and save additional information on the gen, reco, and raw reco jets matched. It will save the data to mlData.txt. This should be transfered to the ProbabilisticPredictions folder. Once there, run cleanData.py as 
```
python cleanData.py --filename=filename
```
where filename is whatever the mlData.txt file was saved as after the transfer. Do not include the .txt ending. The data in the file will be reordered so that within each jet the particles are orderd by descending gen pt. Additionally, it will be saved in a .npy file.

After this, extract the regression and classifcation networks using
```
python extractNetworks.py --filename=filename --regression=regression --classification=classification
```
Here filename is the name of the data file used to train the two networks, regression is the name of the regression network, and classification is the name of the classifcation network. This will store the weights and biases for the networks in a folder called classification and one called regression. The data contained therein should be transfered into the ProbabilisticPredictions folder. From there, dataset.py, and predict.py can be run 
```
python3 dataset.py
python3 predict.py
```
This will make the datasets used for training the correction network. When in the same folder as jetCorrectionQuantile.py, the final trainining can be run as 
```
python jetCorrectionQuantile.py
```
Check the click arguments and make sure you are giving it the correct file name. Once trained, you can check its performance with quantileAlgRawToGenPredictionsGit.ipynb
Additionally, make a dataset for testCluster.py in the ProbabilisticPredictions folder and run in an IDE to examine the clustering algorithm.


View the presentations folder to see presentations on how this project has changed over time.
