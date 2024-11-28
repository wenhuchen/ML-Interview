#include <stdio.h>

__global__ void MatAdd3DKernel(int* A, int* B, int* C) {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int block_offset = block_id * blockDim.x * blockDim.y * blockDim.z;
    int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int tid = block_offset + thread_offset;
    C[tid] = A[tid] + B[tid];
}

void setup(int size, int *h_a, int *h_b, int *h_c, int **d_a, int **d_b, int **d_c){
    // Allocate device memory
    cudaMalloc((void **)d_a, size);
    cudaMalloc((void **)d_b, size);
    cudaMalloc((void **)d_c, size);

    // Copy data from host to device
    // cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind)
    cudaMemcpy(*d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_b, h_b, size, cudaMemcpyHostToDevice);
}


int main() {
    const int n = 512;
    size_t size = n * n * sizeof(int);
    int h_a[n][n], h_b[n][n], h_c[n][n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j){
            h_a[i][j] = i + j;
            h_b[i][j] = i * 2 + j;
        }
    }

    int *d_a, *d_b, *d_c;
    memset(h_c, 0, size);
    setup(size, (int *)h_a, (int *)h_b, (int *)h_c, &d_a, &d_b, &d_c);

    // Running on the 3-D matplus
    dim3 blocks_per_grid(8, 8, 8);
    dim3 threads_per_block(10, 10, 10);
    // So each block contains 8 cores; there are 6 blocks
    MatAdd3DKernel<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the results
    printf("Result:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; ++j){
            printf("%d + %d = %d\n", h_a[i][j], h_b[i][j], h_c[i][j]);
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
