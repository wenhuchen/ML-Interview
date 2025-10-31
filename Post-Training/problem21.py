"""
Problem 21: Gaussian Distribution - KL Divergence, Sampling, and PDF
"""

import numpy as np


def gaussian_kl_divergence(mu1, sigma1, mu2, sigma2):
    """
    KL(N(μ1, σ1²) || N(μ2, σ2²)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²) / (2σ2²) - 0.5
    """
    return np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5


def sample_gaussian(mu, sigma, size=1):
    """Sample from N(μ, σ²)"""
    return np.random.normal(loc=mu, scale=sigma, size=size)


def gaussian_pdf(x, mu, sigma):
    """
    Probability density function (PDF) of N(μ, σ²)
    PDF(x) = (1 / (σ√(2π))) * exp(-0.5 * ((x-μ)/σ)²)
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)


def k1_divergence(samples, q_mu, q_sigma, p_mu, p_sigma):
    """
    Estimate KL divergence using samples from Q
    """
    q = gaussian_pdf(samples, mu=q_mu, sigma=q_sigma)
    p = gaussian_pdf(samples, mu=p_mu, sigma=p_sigma)
    return np.mean(np.log(q / p))

def k2_divergence(samples, q_mu, q_sigma, p_mu, p_sigma):
    """
    Estimate KL divergence using samples from Q
    """
    q = gaussian_pdf(samples, mu=q_mu, sigma=q_sigma)
    p = gaussian_pdf(samples, mu=p_mu, sigma=p_sigma)
    return np.mean(1 / 2 * (np.log(q / p)**2))

def k3_divergence(samples, q_mu, q_sigma, p_mu, p_sigma):
    """
    Estimate KL divergence using samples from Q
    """
    q = gaussian_pdf(samples, mu=q_mu, sigma=q_sigma)
    p = gaussian_pdf(samples, mu=p_mu, sigma=p_sigma)
    return np.mean((p / q - 1) - np.log(p / q))

# Example usage
if __name__ == "__main__":
    # Analytical KL divergence
    q_mu = 1.1
    q_sigma = 1.4
    p_mu = 0.2
    p_sigma = 1.2
    
    kl = gaussian_kl_divergence(mu1=p_mu, sigma1=p_sigma, mu2=q_mu, sigma2=q_sigma)
    print(f"KL(N(0,1) || N(1,4)) = {kl:.6f}")
    
    errors = [[], [], []]
    for _ in range(100):
        # Estimated KL using samples
        samples = sample_gaussian(mu=q_mu, sigma=q_sigma, size=100)
        estimated_kl_k1 = k1_divergence(samples, q_mu=q_mu, q_sigma=q_sigma, p_mu=p_mu, p_sigma=p_sigma)
        estimated_kl_k2 = k2_divergence(samples, q_mu=q_mu, q_sigma=q_sigma, p_mu=p_mu, p_sigma=p_sigma)
        estimated_kl_k3 = k3_divergence(samples, q_mu=q_mu, q_sigma=q_sigma, p_mu=p_mu, p_sigma=p_sigma)

        errors[0].append(abs(estimated_kl_k1 - kl))
        errors[1].append(abs(estimated_kl_k2 - kl))
        errors[2].append(abs(estimated_kl_k3 - kl))

    print(f"Errors = {np.mean(errors[0])}, {np.mean(errors[1])}, {np.mean(errors[2])}")