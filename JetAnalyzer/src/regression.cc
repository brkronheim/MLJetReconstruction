#include <list>
#include <string>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>

#include "MLJetReconstruction/JetAnalyzer/interface/regression.h"

regression::regression(std::string fileName)
{
    std::cout << "Making regression" << std::endl;
    extractMatrix(fileName+"/normInfo1.txt", &normInfo1[0][0], 2);

    /*for(int x = 0; x<58; x++){
        std::cout << *(&normInfo1[0][0] + x) << std::endl;
    }*/

    extractMatrix(fileName+"/normInfo2.txt", &normInfo2[0][0], 2);

    /*for(int x = 0; x<8; x++){
        std::cout << *(&normInfo2[0][0] + x) << std::endl;
    }*/

	extractMatrix(fileName+"/weights_0.txt", &weight0[0][0], 29);
	extractMatrix(fileName+"/biases_0.txt", &bias0[0][0], 1);
	extractMatrix(fileName+"/weights_1.txt", &weight1[0][0], 50);
	extractMatrix(fileName+"/biases_1.txt", &bias1[0][0], 1);
	extractMatrix(fileName+"/weights_2.txt", &weight2[0][0], 50);
	extractMatrix(fileName+"/biases_2.txt", &bias2[0][0], 1);
	extractMatrix(fileName+"/weights_3.txt", &weight3[0][0], 50);
	extractMatrix(fileName+"/biases_3.txt", &bias3[0][0], 1);
	extractMatrix(fileName+"/weights_4.txt", &weight4[0][0], 50);
	extractMatrix(fileName+"/biases_4.txt", &bias4[0][0], 1);
	extractMatrix(fileName+"/weights_5.txt", &weight5[0][0], 50);
	extractMatrix(fileName+"/biases_5.txt", &bias5[0][0], 1);


	weights[0] = &weight0[0][0];
	weights[1] = &weight1[0][0];
	weights[2] = &weight2[0][0];
	weights[3] = &weight3[0][0];
	weights[4] = &weight4[0][0];
	weights[5] = &weight5[0][0];


	biases[0] = &bias0[0][0];
	biases[1] = &bias1[0][0];
	biases[2] = &bias2[0][0];
	biases[3] = &bias3[0][0];
	biases[4] = &bias4[0][0];
	biases[5] = &bias5[0][0];

	/*std::cout << "exit constructor" << std::endl;*/
}

/*regression::~regression()
{
    //dtor
}*/


//Make predictions with the neural net
void regression::predict(float * inputVal, float * outputVal, int dim){


	/*
	float jetPT = *(inputVal + 7) ;
	float jetEta = *(inputVal + 8) ;
	float jetPhi = *(inputVal + 9) ;
	float jetE = *(inputVal + 10) ;
	float particlePT = *(inputVal + 11) ;
	float particleEta = *(inputVal + 12) ;
	float particlePhi = *(inputVal + 13) ;
	float particleE = *(inputVal + 14) ;
	
	float newPx = jetPT * cos(jetPhi) + particlePT * cos(particlePhi);
	float newPy = jetPT * sin(jetPhi) + particlePT * sin(particlePhi);
	float newPz = jetPT * sinh(jetEta) + particlePT * sinh(particleEta);
	
	
	float newPT = sqrt(newPx*newPx + newPy*newPy);
	float newEta = asinh(newPz/newPT);
	float newPhi = asin(newPy/newPT);
	float newE = jetE + particleE;
	
	*(outputVal + 0) = newPT;
	*(outputVal + 1) = newEta;
	*(outputVal + 2) = newPhi;
	*(outputVal + 3) = newE; 
	
	*(outputVal + 0) = *(inputVal + 7)  + *(inputVal + 11) ;
	*(outputVal + 1) = *(inputVal + 8)  + *(inputVal + 12) ;
	*(outputVal + 2) = *(inputVal + 9)  + *(inputVal + 13) ;
	*(outputVal + 3) = *(inputVal + 10)  + *(inputVal + 14) ;*/

    //std::cout << "enter predict" << std::endl;
	
    int inputDim = 29;
    int outputDim = 4;
    int hiddenLayers = 5;
    int layerWdith = 50;

    //std::cout << "A" << std::endl;
    float innerVal1[layerWdith][dim];
	float innerVal2[layerWdith][dim];
	float inputCopy[dim][inputDim];
    //std::cout << "B" << std::endl;
	//Copy input to a new array and normalize the contents
	copyMatrix(inputVal, &inputCopy[0][0], dim, inputDim);
	//std::cout << "C" << std::endl;


	normalize(&normInfo1[0][0],&inputCopy[0][0], inputDim, dim);

	//Start working through the network layers
	//std::cout << "D" << std::endl;
	matrixMult(weights[0], &inputCopy[0][0], &innerVal1[0][0], layerWdith, dim, inputDim);
	//std::cout << "E" << std::endl;
	matrixAdd(&innerVal1[0][0], biases[0], layerWdith, dim);
    //std::cout << "F" << std::endl;
	leakyReLU(&innerVal1[0][0], layerWdith, dim);
	//std::cout << "G" << std::endl;
	copyMatrix(&innerVal1[0][0], &innerVal2[0][0], layerWdith, dim);
    //std::cout << "H" << std::endl;
	for(int x=1; x<hiddenLayers; x++){
		matrixMult(weights[x], &innerVal2[0][0], &innerVal1[0][0], layerWdith, dim, layerWdith);
		matrixAdd(&innerVal1[0][0], biases[x], layerWdith, dim);
		leakyReLU(&innerVal1[0][0], layerWdith, dim);
		copyMatrix(&innerVal1[0][0], &innerVal2[0][0], layerWdith, dim);
	}

	matrixMult(weights[hiddenLayers], &innerVal2[0][0], &innerVal1[0][0], outputDim, dim, layerWdith);
	matrixAdd(&innerVal1[0][0], biases[hiddenLayers], outputDim, dim);

	//Copy output to a new array
	copyMatrix(&innerVal1[0][0], outputVal, outputDim, dim);
	//Round predictions to 0.0 or 1.0
	unnormalize(outputVal, outputDim, dim);
	


}

//Read a file into an array
void regression::extractMatrix(std::string fileName, float * matrix, int dim1){

	std::fstream file (fileName, std::ios_base::in);
	std::string line;


    //std::string line;
    int y =0;
    while (std::getline(file, line))
    {
        std::istringstream linestream(line);
        std:: string value;
        int x = 0;
		while(getline(linestream,value,',')) {
            //std::cout << strtof(value.c_str(), NULL) << std::endl;
			*(matrix + x + (y * dim1)) = strtof(value.c_str(), NULL);
			x++;
		}
		y++;
	}

	//for(int x = 0; x<58; x++){
    //    std::cout << *(matrix + x) << std::endl;
    //}

	/*int y = 0;
	while(getline(file,line)) {
		std::stringstream   linestream(line);
		std::string         value;
		int x = 0;
		while(getline(linestream,value,',')) {
			*(matrix + x + (y * dim1)) = atof(value.c_str());
			x++;
		}
		y++;
	}*/
}

//Multiply two arrays
void regression::matrixMult(float * a, float * b, float * c, int dim1, int dim2, int dim3){
	for(int x=0; x<dim1; x++){
		for(int y=0; y<dim2; y++){
			*(c + y + (x * dim2)) = 0.0;
			for(int z=0; z<dim3; z++){
                //std::cout << x << "," << y << "," << z << std::endl;
				*(c + y + (x * dim2)) += *(a + z + (x * dim3)) * *(b + y + (z * dim2));

			}
		}
	}
}

//Add two arrays
void regression::matrixAdd(float * a, float * b, int dim1, int dim2){
    for(int x=0; x<dim1; x++){
        for(int y=0; y<dim2; y++){


			(*(a + y + (x * dim2))) = *(a + y + (x * dim2)) + *(b + x);

		}
	}
}

//Apply the leakyReLU activation function with a slope of 0.1
void regression::leakyReLU(float * a, int dim1, int dim2){
    for(int x=0; x<dim1; x++){
        for(int y=0; y<dim2; y++){
            if(*(a + y + (x * dim2))<0){
                *(a + y + (x * dim2)) = *(a + y + (x * dim2)) * 0.2;
            } else {
                *(a + y + (x * dim2))= *(a + y + (x * dim2));
			}
		}
	}
}

//Round predictions to 0 or 1
void regression::roundPredictions(float * a, int dim1, int dim2){
    //std::cout << "Round" << std::endl;
	for(int x=0; x<dim1; x++){
        for(int y=0; y<dim2; y++){
			//std::cout << *(a + y + (x * dim2)) << "->";
            if(*(a + y + (x * dim2))<0.5){
                *(a + y + (x * dim2)) = 0.0;
			} else {
                *(a + y + (x * dim2)) = 1.0;
			}
			//std::cout <<  *(a + y + (x * dim2)) << std::endl;
		}
	}
}

//Copy the contents of an array into another
void regression::copyMatrix(float * a, float * b, int dim1, int dim2){
	for(int x=0;x<dim1;x++){
		for(int y =0; y<dim2;y++){
			 *(b + y + (x * dim2)) = *(a + y + (x * dim2));
		}
	}
}

//Normalize the input array
void regression::normalize(float * norm, float * a, int dim1, int dim2){
    //std::cout << "Normalize" << std::endl;
	for(int x=0; x<dim1; x++){
        for(int y=0; y<dim2; y++){
            if(x == 0 || x == 3 || x == 7 || x == 10 || x == 11 || x == 14){
				*(a + y + (x * dim2)) = log(*(a + y + (x * dim2)));
			} else if( x == 4 || x == 5 || x == 6 || x == 15 || x == 16 || x == 17 ){
				*(a + y + (x * dim2)) = cbrt(*(a + y + (x * dim2)));	
			}
			//std::cout << *(a + y + (x * dim2)) << "->";
            *(a + y + (x * dim2)) = (*(a + y + (x * dim2)) - normInfo1[x][0])/normInfo1[x][1];
			//std::cout <<  *(a + y + (x * dim2)) << std::endl;
		}
	}
	//std::cout << "Done Normalizing" << std::endl;
}

//Unnormalize the output array
void regression::unnormalize(float * a, int dim1, int dim2){
	//std::cout << "Unnormalize" << std::endl;
	for(int x=0; x<dim1; x++){
        for(int y=0; y<dim2; y++){
			//std::cout << *(a + y + (x * dim2)) << "->";
            *(a + y + (x * dim2)) = (*(a + y + (x * dim2))*normInfo1[x][1] + normInfo1[x][0]);
			if(x == 0 || x == 3){
				*(a + y + (x * dim2)) = exp(*(a + y + (x * dim2)));
			}
			//std::cout <<  *(a + y + (x * dim2)) << std::endl;
		}
	}
}
