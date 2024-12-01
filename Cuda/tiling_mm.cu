#include <stdio.h>

const int TILE_SIZE = 4;

__global__ void matMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];


    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0;

    for (int i = 0; i < (N + TILE_SIZE - 1) / TILE_SIZE; i++) {
        // Load tiles into shared memory
        if (row < N && (i * TILE_SIZE + threadIdx.x) < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + (i * TILE_SIZE + threadIdx.x)];
        }
        else {
            tileA[threadIdx.y][threadIdx.x] = 0;
        }
        if (col < N && (i * TILE_SIZE + threadIdx.y) < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * N + col];
        }
        else {
            tileB[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        /*if (blockIdx.x == 0 && blockIdx.y == 0) {
            printf("tile A contains: %d %d ---- tile B contains: (%d %d)\n", row, i * TILE_SIZE + threadIdx.x, i * TILE_SIZE + threadIdx.y, col);
        }*/
        __syncthreads();
        /*if (blockIdx.x == 0 && blockIdx.y == 0 & threadIdx.x == 0 & threadIdx.y == 0) {
            printf("i = %d\n\n", i);
        }*/

        // Perform computation for the tile
        for (int j = 0; j < TILE_SIZE; j++) {
            sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
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

    dim3 blocks_per_grid((n + TILE_SIZE - 1 ) / TILE_SIZE, (n + TILE_SIZE - 1 ) / TILE_SIZE);
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);

    cudaEventRecord(start, 0);  // Record the start event
    matMulTiled<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop, 0);   // Record the stop event

    cudaEventSynchronize(stop);  // Wait for the event to complete
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);  // Compute elapsed time
    printf("Kernel execution time: %f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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