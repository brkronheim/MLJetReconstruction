#include <list>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "MLJetReconstruction/JetAnalyzer/interface/classifier.h"

classifier::classifier(std::string fileName)
{
    std::cout << "Making classifier" << std::endl;
    extractMatrix(fileName+"/normInfo1.txt", &normInfo[0][0], 2);

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

}

/*classifier::~classifier()
{
    //dtor
}*/


//Make predictions with the neural net
void classifier::predict(float * inputVal, float * outputVal, int dim){
    int inputDim = 29;
    int outputDim = 1;
    int hiddenLayers = 5;
    int layerWdith = 50;


    float innerVal1[layerWdith][dim];
	float innerVal2[layerWdith][dim];
	float inputCopy[dim][inputDim];

	//Copy input to a new array and normalize the contents
	copyMatrix(inputVal, &inputCopy[0][0], dim, inputDim);
	normalize(&inputCopy[0][0], inputDim, dim);

	//Start working through the network layers
	matrixMult(weights[0], &inputCopy[0][0], &innerVal1[0][0], layerWdith, dim, inputDim);
	matrixAdd(&innerVal1[0][0], biases[0], layerWdith, dim);

	leakyReLU(&innerVal1[0][0], layerWdith, dim);
	copyMatrix(&innerVal1[0][0], &innerVal2[0][0], layerWdith, dim);

	for(int x=1; x<hiddenLayers; x++){
		matrixMult(weights[x], &innerVal2[0][0], &innerVal1[0][0], layerWdith, dim, layerWdith);
		matrixAdd(&innerVal1[0][0], biases[x], layerWdith, dim);
		leakyReLU(&innerVal1[0][0], layerWdith, dim);
		copyMatrix(&innerVal1[0][0], &innerVal2[0][0], layerWdith, dim);
	}

	matrixMult(weights[hiddenLayers], &innerVal2[0][0], &innerVal1[0][0], outputDim, dim, layerWdith);
	matrixAdd(&innerVal1[0][0], biases[hiddenLayers], outputDim, dim);
	sigmoid(&innerVal1[0][0], outputDim, dim);

	//Copy output to a new array
	copyMatrix(&innerVal1[0][0], outputVal, outputDim, dim);
	//Round predictions to 0.0 or 1.0
	roundPredictions(outputVal, outputDim, dim);

}

//Read a file into an array
void classifier::extractMatrix(std::string fileName, float * matrix, int dim1){

	std::fstream file (fileName, std::ios_base::in);
	std::string line;


    //std::string line;
    int y =0;
    while (std::getline(file, line))
    {
        std::istringstream linestream(line);
        std::string value;
        int x = 0;
		while(getline(linestream,value,',')) {
			*(matrix + x + (y * dim1)) = atof(value.c_str());
			x++;
		}
		y++;
	}

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
void classifier::matrixMult(float * a, float * b, float * c, int dim1, int dim2, int dim3){
	for(int x=0; x<dim1; x++){
		for(int y=0; y<dim2; y++){
			*(c + y + (x * dim2)) = 0.0;
			for(int z=0; z<dim3; z++){
                *(c + y + (x * dim2)) += *(a + z + (x * dim3)) * *(b + y + (z * dim2));

			}
		}
	}
}

//Add two arrays
void classifier::matrixAdd(float * a, float * b, int dim1, int dim2){
    for(int x=0; x<dim1; x++){
        for(int y=0; y<dim2; y++){


			(*(a + y + (x * dim2))) = *(a + y + (x * dim2)) + *(b + x);

		}
	}
}

//Apply the leakyReLU activation function with a slope of 0.1
void classifier::leakyReLU(float * a, int dim1, int dim2){
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

//Apply the sigmoid activation function
void classifier::sigmoid(float * a, int dim1, int dim2){
	//std::cout << "Sigmoid" << std::endl;
	for(int x=0; x<dim1; x++){
        for(int y=0; y<dim2; y++){
			//std::cout << *(a + y + (x * dim2)) << "->";
			if(*(a + y + (x * dim2)) <-1000){
                *(a + y + (x * dim2)) = 0.0;
			} else if(*(a + y + (x * dim2)) > 1000){
                *(a + y + (x * dim2)) = 1.0;
			} else {
                *(a + y + (x * dim2)) = 1/(1 + exp(- *(a + y + (x * dim2))));
			}
			//std::cout <<  *(a + y + (x * dim2)) << std::endl;
		}
	}
}

//Round predictions to 0 or 1
void classifier::roundPredictions(float * a, int dim1, int dim2){
    //std::cout << "Round" << std::endl;
	for(int x=0; x<dim1; x++){
        for(int y=0; y<dim2; y++){
			//std::cout << *(a + y + (x * dim2)) << "->";
            if(*(a + y + (x * dim2))<0.4){
                *(a + y + (x * dim2)) = 0.0;
			} else {
                *(a + y + (x * dim2)) = 1.0;
			}
			//std::cout <<  *(a + y + (x * dim2)) << std::endl;
		}
	}
}

//Copy the contents of an array into another
void classifier::copyMatrix(float * a, float * b, int dim1, int dim2){
	for(int x=0;x<dim1;x++){
		for(int y =0; y<dim2;y++){
			 *(b + y + (x * dim2)) = *(a + y + (x * dim2));
		}
	}
}

//Normalize the input array
void classifier::normalize(float * a, int dim1, int dim2){
	//std::cout << "Normalize" << std::endl;
	for(int x=0; x<dim1; x++){
        for(int y=0; y<dim2; y++){
			if(x == 0 || x == 3 || x == 7 || x == 10 || x == 11 || x == 14){
				*(a + y + (x * dim2)) = log(*(a + y + (x * dim2)));
			} else if( x == 4 || x == 5 || x == 6 || x == 15 || x == 16 || x == 17 ){
				*(a + y + (x * dim2)) = cbrt(*(a + y + (x * dim2)));	
			}
			//std::cout << *(a + y + (x * dim2)) << "->";
            *(a + y + (x * dim2)) = (*(a + y + (x * dim2)) - normInfo[x][0])/normInfo[x][1];
			//std::cout <<  *(a + y + (x * dim2)) << std::endl;
			
		}
	}
	//std::cout << std::endl;
}
