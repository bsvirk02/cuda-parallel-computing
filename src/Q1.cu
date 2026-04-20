#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "com2039.hpp"  // Contains kernel declarations and constants

/// Error checking macro for CUDA API calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/// Function to load binary floating-point data from a .pbin file
size_t loadSamples(const char* path_to_data_points_file, float** ptr) {
    FILE* file = fopen(path_to_data_points_file, "rb");
    if (!file) {
        printf("Error opening file: %s\n", path_to_data_points_file);
        exit(1);
    }

    fseek(file, 0, SEEK_END);                      // Seek to end to get size
    long size_read = ftell(file);                  // Get total bytes
    rewind(file);                                  // Go back to start

    if (size_read < 0) {
        printf("Error determining file size.\n");
        fclose(file);
        exit(1);
    }

    size_t len_array = size_read / sizeof(float);  // Calculate number of float elements
    printf("Read: %ld bytes = %zu elements.\n", size_read, len_array);

    char* memblock = new char[size_read];
    if (fread(memblock, 1, size_read, file) != (size_t)size_read) {
        printf("Error reading file contents.\n");
        fclose(file);
        exit(1);
    }

    fclose(file);
    *ptr = (float*)memblock;  // Cast char buffer to float pointer

    printf("Correctly loaded %s\n", path_to_data_points_file);
    return len_array;
}

/// CUDA Kernel to compute minimum value using parallel reduction
__global__ void minReduceKernel(float* d_in, int len) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float localMin = FLT_MAX; // Identity element for min
    if (idx < len)
        localMin = d_in[idx];
    if (idx + blockDim.x < len)
        localMin = fminf(localMin, d_in[idx + blockDim.x]);

    sdata[tid] = localMin;     // Store in shared memory
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    // Write per-block result to global memory
    if (tid == 0)
        d_in[blockIdx.x] = sdata[0];
}

/// Wrapper function to compute the minimum value from host
float findMinValue(float* samples_h, size_t numSamples) {
    float* d_data;
    size_t size = numSamples * sizeof(float);
    gpuErrchk(cudaMalloc(&d_data, size));
    gpuErrchk(cudaMemcpy(d_data, samples_h, size, cudaMemcpyHostToDevice));

    size_t curr_len = numSamples;
    while (curr_len > 1) {
        size_t threads = 1024;
        size_t blocks = (curr_len + threads * 2 - 1) / (threads * 2);
        minReduceKernel<<<blocks, threads, threads * sizeof(float)>>>(d_data, curr_len);
        gpuErrchk(cudaDeviceSynchronize());
        curr_len = blocks;  // Update size for next reduction round
    }

    float min_val;
    gpuErrchk(cudaMemcpy(&min_val, d_data, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
    return min_val;
}

/// CUDA Kernel to compute maximum value using parallel reduction
__global__ void maxReduceKernel(float* d_in, int len) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float localMax = -FLT_MAX; // Identity element for max
    if (idx < len)
        localMax = d_in[idx];
    if (idx + blockDim.x < len)
        localMax = fmaxf(localMax, d_in[idx + blockDim.x]);

    sdata[tid] = localMax;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    // Write per-block result to global memory
    if (tid == 0)
        d_in[blockIdx.x] = sdata[0];
}

/// Wrapper function to compute the maximum value from host
float findMaxValue(float* samples_h, size_t numSamples) {
    float* d_data;
    size_t size = numSamples * sizeof(float);
    gpuErrchk(cudaMalloc(&d_data, size));
    gpuErrchk(cudaMemcpy(d_data, samples_h, size, cudaMemcpyHostToDevice));

    size_t curr_len = numSamples;
    while (curr_len > 1) {
        size_t threads = 1024;
        size_t blocks = (curr_len + threads * 2 - 1) / (threads * 2);
        maxReduceKernel<<<blocks, threads, threads * sizeof(float)>>>(d_data, curr_len);
        gpuErrchk(cudaDeviceSynchronize());
        curr_len = blocks;
    }

    float max_val;
    gpuErrchk(cudaMemcpy(&max_val, d_data, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
    return max_val;
}

/// CUDA Kernel to generate a histogram with 512 bins
__global__ void histogramKernel512(float* d_in, unsigned int* hist, size_t len_array, float min_value, float bin_width) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len_array) return;

    float val = d_in[idx];
    int bin = (int)((val - min_value) / bin_width);  // Compute bin index
    if (bin >= NUM_BINS) bin = NUM_BINS - 1;         // Clamp to last bin (inclusive upper bound)
    if (bin < 0) bin = 0;

    atomicAdd(&(hist[bin]), 1);  // Atomic update to avoid race conditions
}

/// Wrapper to manage histogram kernel launch and memory operations
void histogram512(float* samples_h, size_t len_array, unsigned int** hist_h, float min_value, float max_value) {
    float* d_in;
    unsigned int* d_hist;
    size_t size_data = len_array * sizeof(float);
    size_t size_hist = NUM_BINS * sizeof(unsigned int);

    // Allocate and copy input data
    gpuErrchk(cudaMalloc(&d_in, size_data));
    gpuErrchk(cudaMemcpy(d_in, samples_h, size_data, cudaMemcpyHostToDevice));

    // Allocate and initialize histogram on device
    gpuErrchk(cudaMalloc(&d_hist, size_hist));
    gpuErrchk(cudaMemset(d_hist, 0, size_hist));

    // Calculate bin width from range
    float bin_width = (max_value - min_value) / NUM_BINS;

    // Launch histogram kernel
    size_t threads = 1024;
    size_t blocks = (len_array + threads - 1) / threads;
    histogramKernel512<<<blocks, threads>>>(d_in, d_hist, len_array, min_value, bin_width);
    gpuErrchk(cudaDeviceSynchronize());

    // Copy histogram back to host
    gpuErrchk(cudaMemcpy(*hist_h, d_hist, size_hist, cudaMemcpyDeviceToHost));

    // Clean up device memory
    cudaFree(d_in);
    cudaFree(d_hist);
}
