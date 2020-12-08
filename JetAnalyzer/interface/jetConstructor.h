#ifndef JETCONSTRUCTOR_H
#define JETCONSTRUCTOR_H

#include "classifier.h"
#include "regression.h"
#include <string>

class jetConstructor {
    public:
        jetConstructor();
        //virtual ~jetConstructor();
        int run(float ** particles, int particleCount, float ** outputJets);

    protected:

    private:
        float dR2(float eta1, float eta2, float phi1, float phi2);
        float dPhi(float phi1, float phi2);


        std::string strRegression = "/home/cmsusr/CMSSW_10_6_8_patch1/src/MLJetReconstruction/JetAnalyzer/data/regression";
	std::string strClassifier = "/home/cmsusr/CMSSW_10_6_8_patch1/src/MLJetReconstruction/JetAnalyzer/data/classification";
	
        regression regressionNet = regression(strRegression);
	classifier classifierNet = classifier(strClassifier);

};

#endif // JETCONSTRUCTOR_H
