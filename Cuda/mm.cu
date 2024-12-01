#include <stdio.h>

__global__ void matmul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        C[row * N + col] = 0;
        for (int k = 0; k < N; k++) {
            C[row * N + col] += A[row * N + k] * B[k * N + col];
        }
    }
}

void setup(int size, float *h_a, float *h_b, float *h_c, float **d_a, float **d_b, float **d_c){
    // Allocate device memory
    cudaMalloc((void **)d_a, size);
    cudaMalloc((void **)d_b, size);
    cudaMalloc((void **)d_c, size);

    // Copy data from host to device
    cudaMemcpy(*d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_b, h_b, size, cudaMemcpyHostToDevice);
}

int main() {
    int n = 512; // Size of the input array
    size_t size = n * n * sizeof(float);
    const int BLOCK_SIZE = 32;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float h_a[n][n], h_b[n][n], h_c[n][n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j){
            h_a[i][j] = 1;
            h_b[i][j] = 1;
        }
    }

    float *d_a, *d_b, *d_c;
    memset(h_c, 0, size);
    setup(size, (float *)h_a, (float *)h_b, (float *)h_c, &d_a, &d_b, &d_c);

    dim3 blocks_per_grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);

    cudaEventRecord(start, 0);  // Record the start event
    matmul<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop, 0);   // Record the stop event

    cudaEventSynchronize(stop);  // Wait for the event to complete
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);  // Compute elapsed time
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the results
    printf("Result:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; ++j){
            printf("%f ", h_c[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}