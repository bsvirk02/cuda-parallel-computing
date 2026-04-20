
//============================================================================
// Name        : com2039.hpp
// Author      :
// Version     :
// Copyright   :
// Description : COM2039 Histogram Coursework
//============================================================================

#ifndef COM2039_HPP_
#define COM2039_HPP_

#include <iostream>
#include <fstream>
#include <cfloat>

#include "cuda_runtime.h"

using namespace std;

const size_t BLOCK_SIZE = 1024;
const size_t NUM_BINS = 512;

// Find Maximum
__global__ void maxReduceKernel(float *d_in, int lenArray);
float findMaxValue(float* samples_h, size_t numSamples);

// Find Minimum
__global__ void minReduceKernel(float *d_in, int lenArray);
float findMinValue(float* samples_h, size_t numSamples);

// Histogram Kernel
__global__ void histogramKernel512(float* d_in, unsigned int *hist, size_t lenArray, float min_value, float div);

// Histogram Wrapper
void histogram512(float* samples_h, size_t numSamples, unsigned int **hist_h, float minValue, float maxValue);

// Load files
size_t loadSamples(const char* path_to_data_points_file, float** ptr );

#endif /* COM2039_HPP_ */
