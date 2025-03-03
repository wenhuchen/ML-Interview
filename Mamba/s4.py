import numpy as np
from scipy import linalg

class S4:
    def __init__(self):
        # Initialize dimensions
        self.N = 64  # State dimension
        self.dt = 0.01  # Discretization step size
        self.d_model = 16  # Output dimension
        
        # Create state space matrices
        self.A = self._hippo_matrix()
        self.B = self._init_B()
        self.C = self._init_C()
        
        # Discretize the system
        self.Ad = self._discretize()
        self.Bd = self._discretize_B()
        
    def _hippo_matrix(self):
        """
        Construct HiPPO-LegS matrix of size N x N.
        This is the original HiPPO matrix without scaling.
        """
        
        N = self.N
        A = np.zeros((N, N))
        
        # Construct the HiPPO-LegS matrix
        for k in range(N):
            for j in range(N):
                if k < j:
                    A[k, j] = 0
                elif k == j:
                    A[k, j] = k + 1
                else:
                    A[k, j] = -np.sqrt((2*k + 1)*(2*j + 1))
                    
        return A
        
    def _discretize(self):
        """
        Discretize the continuous-time HiPPO matrix using zero-order hold (ZOH).
        Returns the discrete-time state matrix Ad.
        
        ZOH: Ad = exp(A*dt)
        """
        
        # Zero-order hold discretization
        Ad = linalg.expm(self.A * self.dt)
        return Ad
        
    def _init_B(self):
        """
        Initialize the input matrix B.
        For S4, B is typically chosen as a random vector.
        """
        
        # Initialize B as a random normal vector
        B = np.random.randn(self.N, 1)
        # Scale the norm to 1
        B = B / np.sqrt(np.sum(B**2))
        return B
        
    def _init_C(self):
        """
        Initialize the output matrix C.
        Projects the N-dimensional state to d_model dimensions.
        """
        
        # Initialize C as a random normal matrix
        C = np.random.randn(self.d_model, self.N) / np.sqrt(self.N)
        return C
        
    def _discretize_B(self):
        """
        Discretize the input matrix B using zero-order hold (ZOH).
        For a ZOH system: Bd = A^(-1)(Ad - I)B
        """
        
        # Compute Bd using the ZOH formula
        I = np.eye(self.N)
        Bd = linalg.solve(self.A, (self.Ad - I) @ self.B)
        return Bd
        
    def forward(self, u, x0=None):
        """
        Forward pass through the S4 system.
        Args:
            u: Input sequence of shape (L, batch_size, 1)
            x0: Initial state of shape (batch_size, N). If None, starts from zero.
        Returns:
            y: Output sequence of shape (L, batch_size, d_model)
        """
        
        # Get sequence length and batch size
        L, batch_size, _ = u.shape
        
        # Initialize state
        if x0 is None:
            x = np.zeros((batch_size, self.N))
        else:
            x = x0
            
        # Initialize output array
        y = np.zeros((L, batch_size, self.d_model))
        
        # Step through the sequence
        for i in range(L):
            # Compute output: y[k] = Cx[k]
            y[i] = x @ self.C.T
            
            # Update state: x[k+1] = Adx[k] + Bdu[k]
            x = x @ self.Ad.T + u[i] @ self.Bd.T
            
        return y
        

def main():
    s4 = S4()
    u = np.random.randn(100, 1, 1)
    y = s4.forward(u)
    print(y)
    print(y.shape)

if __name__ == "__main__":
    main()