#include <stdio.h>

__global__ void blockSumKernel(float *input, float *output, int n) {
    // Allocate shared memory
    __shared__ float sharedData[256];
    __shared__ int counter;

    // Initialize the shared counter to 0
    if (threadIdx.x == 0) {
        counter = 0;
    }
    __syncthreads(); // Ensure all threads see the initialized value

    // Calculate thread and block indices
    int tid = threadIdx.x; // Local thread ID within the block
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global index

    // Load data from global memory to shared memory
    sharedData[tid] = (idx < n) ? input[idx] : 0.0f; // Check bounds
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
            atomicAdd(&counter, 1);
        }
        __syncthreads();
    }

    // Write the result of the reduction for this block to global memory
    if (tid == 0) {
        printf("Total number of addition: %d\n", counter);
        output[blockIdx.x] = sharedData[0];
    }
}

int main() {
    int n = 1024; // Size of the input array
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate and initialize host memory
    printf("Number of blocks: %d\n\n", blocksPerGrid);
    float *h_input = (float *)malloc(n * sizeof(float));
    float *h_output = (float *)malloc(blocksPerGrid * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_input[i] = i; // Example: all elements are 1.0
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, blocksPerGrid * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    blockSumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

    // Copy results back to host
    cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum the partial results on the host
    float totalSum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        totalSum += h_output[i];
    }

    printf("Sum: %f\n", totalSum);

    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
