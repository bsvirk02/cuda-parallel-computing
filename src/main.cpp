//============================================================================
// Name        : main.cpp
// Author      :
// Version     :
// Copyright   :
// Description : COM2039 Histogram Coursework
//============================================================================

#include "com2039.hpp"
#include <cmath>

using namespace std;

int main(int argc, char* argv[]) {
	if (argc != 2 ){
		std::cout << "Remember to pass the path to data_points.txt" << std::endl;
		exit(1);
	}

	// Load dataset from file
	float *samples_h;
	size_t numSamples = loadSamples(argv[1], &samples_h);

	std::cout << "length of vector " << numSamples << std::endl;

	// Find maximum and minimum values
	float maxValue = findMaxValue(samples_h, numSamples);
	std::cout << "GPU Max: " << maxValue << std::endl;

	float minValue = findMinValue(samples_h, numSamples);
	std::cout << "GPU Min: " << minValue << std::endl;

	// Find histogram
	unsigned int *hist_h = new unsigned int[NUM_BINS];

	histogram512(samples_h, numSamples, &hist_h, minValue,  maxValue);
	unsigned long int counter = 0;
	for (int j = 0; j < NUM_BINS ; j++){
		std::cout<< "Bin[" << j <<"]: " << hist_h[j] << std::endl;
		counter += hist_h[j];
	}

	std::cout << "Total number of elements in histogram: " << counter << std::endl;

	// Free memory
	free(samples_h);
	delete[] hist_h;

   return 0;
}
