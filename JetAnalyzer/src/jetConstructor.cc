#include "MLJetReconstruction/JetAnalyzer/interface/jetConstructor.h"
#include "MLJetReconstruction/JetAnalyzer/interface/regression.h"
#include "MLJetReconstruction/JetAnalyzer/interface/classifier.h"
//#include "classifier.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


jetConstructor::jetConstructor() {
    std::cout << "Making jetConstructor" << std::endl;
}


int jetConstructor::run(float ** particles, int particleCount, float ** outputJets){ //
    std::cout << "Number of particles given:" << particleCount << std::endl;
	int jetCount = 0;
    //int matchedParticles = 0;
    int remainingParticles = particleCount;
    float inputVector[29];
    float regressionOutput[4];
    float classificationOutput[1];
    float * rootParticle = nullptr;
    float * currentParticle = nullptr;
    float * currentJet = nullptr;

    /*for(int x = 0; x<particleCount; x++){
        for(int y = 0; y<18; y++){
            std::cout << *( *(particles+x) + y) << ", ";
        }
        std::cout << std::endl;
    }*/

	bool notEnd = true;
    while(notEnd && remainingParticles > 0){
        rootParticle = *(particles + jetCount);
        currentJet = *(particles + jetCount);
        //remainingParticles-=1;
        //std::copy(inputVector, inputVector + 5, rootParticle);
		if(*(rootParticle)<5){
			remainingParticles-=1;
			//*(outputJets + jetCount) = currentJet;
			//jetCount++;
			notEnd = false;
			continue;
		}

        //Extract the 4-vec and vertex information from the root Particle and give it to inputVector
        inputVector[0] = *(rootParticle+0);
        inputVector[1] = *(rootParticle+1);
        inputVector[2] = *(rootParticle+2);
        inputVector[3] = *(rootParticle+3);
        inputVector[4] = *(rootParticle+4);
        inputVector[5] = *(rootParticle+5);
        inputVector[6] = *(rootParticle+6);
        inputVector[7] = 0.01;
        inputVector[8] = 0.0;
        inputVector[9] = 0.0;
        inputVector[10] = 0.01;


        int particlesAdded=0;
        //std::cout << std::endl;
        for(int x = 0; x<remainingParticles; x++){
			if(false && jetCount + x < 100){
				std::cout << "Current particle number"<< jetCount + x << std::endl;
            }
			currentParticle = *(particles + jetCount + x);

            //Extract necesary information from currentParticle, add it to inputVector
            //std::copy(inputVector+11, inputVector + 29, currentParticle);
            std::copy(currentParticle, currentParticle + 18, inputVector+11);
            

            float dR2Val = dR2(*(inputVector+8),*(inputVector+12),*(inputVector+9),*(inputVector+13));
            //std::cout << dR2Val << std::endl;
            if(true || dR2Val < 2){
                classifierNet.predict(&inputVector[0], &classificationOutput[0], 1);
                if(x==0 || classificationOutput[0]==1.0){
                    
					for(int y = 7; y<15; y++){
						std::cout << *( inputVector + y) << ", ";
					}
					std::cout << std::endl;
					
					regressionNet.predict(&inputVector[0], &regressionOutput[0], 1);
                    for(int y = 0; y<4; y++){
                        std::cout << regressionOutput[y] << ", ";
                    }
                    std::cout << std::endl;
					std::cout << std::endl;
                    *(currentJet+0) = regressionOutput[0];
                    *(currentJet+1) = regressionOutput[1];
                    *(currentJet+2) = regressionOutput[2];
                    *(currentJet+3) = regressionOutput[3];

                    //Update input vector based on regression output
                    inputVector[7] = regressionOutput[0];
                    inputVector[8] = regressionOutput[1];
                    inputVector[9] = regressionOutput[2];
                    inputVector[10] = regressionOutput[3];

                    particlesAdded++;
                } else {
                    //Shift the rejected particle back
                    *(particles + jetCount + 1 + x - particlesAdded) = *(particles + jetCount + x);
                }

            } else {
                //Shift the rejected particle back
                *(particles + jetCount + 1 + x - particlesAdded) = *(particles + jetCount + x);
            }

        }
        remainingParticles-=particlesAdded;
        *(outputJets + jetCount) = currentJet;
		jetCount++;
    }
    return(jetCount);

}

float jetConstructor::dR2(float eta1, float eta2, float phi1, float phi2) {
    float deta = eta1 - eta2;
    float dphi = dPhi(phi1, phi2);
    return deta*deta + dphi*dphi;
}
float jetConstructor::dPhi(float phi1, float phi2) {
    float raw_dphi = phi1 - phi2;
    if (fabs(raw_dphi) < 3.14159265) return raw_dphi;
    float region = std::round(raw_dphi / (2.*3.14159265));
    return raw_dphi - 2.*3.14159265*region;
}


/*
Input vector order:
firstPT
firstEta
firstPhi
firstE
firstVx
firstVy
firstVz
jetPT
jetEta
jetPhi
jetE
particlePT
particleEta
particlePhi
particleE
particleVx
partilceVy
particleVz
pdgId(11)

Particle vector order:
particlePT
particleEta
particlePhi
particleE
particleVx
partilceVy
particleVz
pdgId(11)


*/
