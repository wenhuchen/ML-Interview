#include <stdio.h>

__global__ void vectorAdd(const int *a, const int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void setup(int size, int h_a[], int h_b[], int h_c[], int **d_a, int **d_b, int **d_c){
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
    size_t size = n * sizeof(int);
    int h_a[n], h_b[n], h_c[n];
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    int *d_a, *d_b, *d_c;
    memset(h_c, 0, size);
    setup(size, h_a, h_b, h_c, &d_a, &d_b, &d_c);

    // Launch the kernel; this should be less than the cores/SM, so it's less than 1024
    int threadsPerBlock = 256;
    // The additional (threadsPerBlock - 1)) is not covering the remainder
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("Number of Blocks: %d\n\n", blocksPerGrid);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the results
    printf("Result:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
