import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.tools import make_default_context
import numpy as np
import skcuda.linalg as linalg
import skcuda.misc as misc

# Initialize cuBLAS
linalg.init()

# Create matrices
A = np.random.rand(3, 3).astype(np.float32)
B = np.random.rand(3, 3).astype(np.float32)

# Allocate GPU memory and copy data
A_gpu = cuda.mem_alloc(A.nbytes)
B_gpu = cuda.mem_alloc(B.nbytes)
C_gpu = cuda.mem_alloc(A.nbytes)  # Result matrix
cuda.memcpy_htod(A_gpu, A)
cuda.memcpy_htod(B_gpu, B)

# Perform matrix multiplication using cuBLAS
C_gpu = linalg.dot(A, B)

# Copy result back to host
C = np.empty_like(A)
cuda.memcpy_dtoh(C, C_gpu)
print(C)