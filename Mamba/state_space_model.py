import numpy as np
import matplotlib.pyplot as plt


def verify_convolution_theorem():
    # Define two example signals
    x = np.random.randn(100)  # Generate 100 random samples from normal distribution
    h = np.random.uniform(-2, 2, 50)  # Generate 50 random samples uniformly between -2 and 2

    # Perform convolution in the time domain
    y_time = np.convolve(x, h)

    # To apply the convolution theorem, we need to zero-pad x and h to the length of the convolution result.
    N = len(y_time)

    # Compute the FFT of both signals with zero padding
    X = np.fft.fft(x, n=N)
    H = np.fft.fft(h, n=N)

    # Multiply the Fourier transforms (pointwise multiplication)
    Y_freq = X * H

    # Compute the inverse FFT to get the time-domain result
    y_freq = np.fft.ifft(Y_freq)

    # Print the results
    print("Delta between two results:")
    print(y_time - y_freq.real)

    # Plot both results to visually compare
    plt.figure(figsize=(10, 6))
    plt.stem(range(len(y_time)), y_time, linefmt='b-', markerfmt='bo', basefmt=" ", label="Time domain convolution")
    plt.stem(range(len(y_freq)), np.round(y_freq.real, decimals=5), linefmt='r--', markerfmt='ro', basefmt=" ", label="Frequency domain IFFT")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.title("Convolution in Time Domain vs. Multiplication in Frequency Domain")
    plt.legend()
    plt.savefig("convolution_theorem.png")


def state_space_model():
    # Define system matrices for a 2-state system with 1 input and 1 output
    A = np.array([[0.9, 0.1],
                [0.0, 0.8]])
    B = np.array([[0.1],
                [0.2]])
    C = np.array([1, 0])
    D = np.array([0])

    # Simulation parameters
    num_steps = 50                   # Number of time steps
    x = np.zeros((2, num_steps + 1))   # State vector: 2 states
    y = np.zeros(num_steps + 1)        # Output vector
    u = np.ones(num_steps)             # Constant input u[k] = 1 for all k

    # Initial condition for the states
    x[:, 0] = [0, 0]

    # Simulate the state-space model over time
    for k in range(num_steps):
        # Update states: x[k+1] = A * x[k] + B * u[k]
        x[:, k + 1] = A @ x[:, k] + B.flatten() * u[k]
        # Compute output: y[k] = C * x[k] + D * u[k]
        y[k] = C @ x[:, k] + D * u[k]

    # Compute the final output at the last time step
    y[num_steps] = C @ x[:, num_steps] + D * u[-1]

    # Plot the state trajectories and system output
    time = np.arange(num_steps + 1)

    plt.figure(figsize=(12, 5))

    # Plot states x1 and x2 over time
    plt.subplot(1, 2, 1)
    plt.plot(time, x[0, :], label='State x1')
    plt.plot(time, x[1, :], label='State x2')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.title('State Trajectories')
    plt.legend()

    # Plot output y over time
    plt.subplot(1, 2, 2)
    plt.plot(time, y, 'r', label='Output y')
    plt.xlabel('Time Step')
    plt.ylabel('Output Value')
    plt.title('System Output')
    plt.legend()

    plt.tight_layout()
    plt.savefig("state_space_model.png")


def main():
    verify_convolution_theorem()
    state_space_model()

if __name__ == "__main__":
    main()