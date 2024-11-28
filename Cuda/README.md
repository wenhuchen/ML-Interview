## Adding two vectors:

```
nvcc  -o vector_add vector_add.cu
./vector_add
```

## Adding two matrices

```
nvcc  -o matrix_add matrix_add.cu
./matrix_add
```

## Summing matrix row-wise

```
nvcc  -o shared_memory shared_memory.cu
./shared_memory
```

## Kernel Function

```
MatAdd3DKernel<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c);
```
This function will invoke different grids of threads to run. The total number of threads should be larger than the total size of the matrix. Otherwise, some of the output will become zero.

## Shared Function

```
__shared__ float sharedData[256];
__shared__ int counter;
```
This variable will be accessed by all the threads in the block