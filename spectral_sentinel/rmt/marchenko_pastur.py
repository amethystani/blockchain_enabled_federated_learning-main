"""
Marchenko-Pastur Law Implementation

Theoretical foundation for detecting Byzantine gradients via spectral analysis.
The MP law describes the limiting distribution of eigenvalues for random matrices.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional
import warnings


class MarchenkoPasturLaw:
    """
    Marchenko-Pastur distribution for random matrix theory.
    
    For a data matrix X ∈ R^{n×d} with i.i.d. entries from distribution with
    variance σ², the empirical spectral distribution of (1/n)X^T X converges
    to the Marchenko-Pastur law as n,d → ∞ with aspect ratio γ = n/d.
    
    The MP density is:
        ρ(λ) = (1/2πσ²λγ) * √[(λ_+ - λ)(λ - λ_-)]
    where λ_± = σ²(1 ± √γ)²
    """
    
    def __init__(self, aspect_ratio: float, variance: float = 1.0):
        """
        Initialize MP law.
        
        Args:
            aspect_ratio: γ = n/d (number of samples / dimension)
            variance: σ² of the underlying distribution
        """
        self.gamma = aspect_ratio
        self.sigma_sq = variance
        
        # Compute support bounds
        sqrt_gamma = np.sqrt(self.gamma)
        self.lambda_minus = self.sigma_sq * (1 - sqrt_gamma) ** 2
        self.lambda_plus = self.sigma_sq * (1 + sqrt_gamma) ** 2
        
    def density(self, lambda_vals: np.ndarray) -> np.ndarray:
        """
        Compute Marchenko-Pastur density.
        
        Args:
            lambda_vals: Eigenvalues to evaluate density at
            
        Returns:
            MP density values
        """
        lambda_vals = np.asarray(lambda_vals)
        density = np.zeros_like(lambda_vals)
        
        # Only non-zero in support [λ_-, λ_+]
        mask = (lambda_vals >= self.lambda_minus) & (lambda_vals <= self.lambda_plus)
        
        if np.any(mask):
            lam = lambda_vals[mask]
            sqrt_term = np.sqrt(
                (self.lambda_plus - lam) * (lam - self.lambda_minus)
            )
            density[mask] = sqrt_term / (2 * np.pi * self.sigma_sq * lam * self.gamma)
        
        return density
    
    def cdf(self, lambda_vals: np.ndarray) -> np.ndarray:
        """
        Compute cumulative distribution function (numerical integration).
        
        Args:
            lambda_vals: Points to evaluate CDF
            
        Returns:
            CDF values
        """
        lambda_vals = np.asarray(lambda_vals)
        cdf_vals = np.zeros_like(lambda_vals)
        
        for i, lam in enumerate(lambda_vals):
            if lam <= self.lambda_minus:
                cdf_vals[i] = 0.0
            elif lam >= self.lambda_plus:
                cdf_vals[i] = 1.0
            else:
                # Numerical integration
                x = np.linspace(self.lambda_minus, lam, 1000)
                y = self.density(x)
                cdf_vals[i] = np.trapz(y, x)
        
        return cdf_vals
    
    def sample(self, n_samples: int, n_features: int) -> np.ndarray:
        """
        Generate random matrix whose eigenvalues follow MP law.
        
        Args:
            n_samples: Number of samples (rows)
            n_features: Number of features (columns)
            
        Returns:
            Random matrix n_samples × n_features
        """
        # Generate random Gaussian matrix
        X = np.random.randn(n_samples, n_features) * np.sqrt(self.sigma_sq)
        return X
    
    def get_support(self) -> Tuple[float, float]:
        """Get support [λ_-, λ_+] of the MP distribution."""
        return self.lambda_minus, self.lambda_plus
    
    def is_in_support(self, lambda_val: float, tolerance: float = 1e-6) -> bool:
        """
        Check if eigenvalue is in MP support.
        
        Args:
            lambda_val: Eigenvalue to check
            tolerance: Numerical tolerance
            
        Returns:
            True if in support
        """
        return (self.lambda_minus - tolerance <= lambda_val <= 
                self.lambda_plus + tolerance)
    
    @staticmethod
    def estimate_parameters(eigenvalues: np.ndarray, 
                           n_samples: int,
                           n_features: int) -> Tuple[float, float]:
        """
        Estimate MP parameters from empirical eigenvalues.
        
        Args:
            eigenvalues: Observed eigenvalues
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            (aspect_ratio, variance) estimates
        """
        gamma = n_samples / n_features
        
        # Estimate variance from mean eigenvalue
        # E[λ] = σ²(1 + 1/γ) for MP distribution
        mean_lambda = np.mean(eigenvalues)
        sigma_sq = mean_lambda / (1 + 1/gamma)
        
        return gamma, sigma_sq
    
    def ks_test(self, eigenvalues: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for MP conformance.
        
        Tests null hypothesis: eigenvalues follow MP distribution
        
        Args:
            eigenvalues: Empirical eigenvalues
            
        Returns:
            (statistic, p_value)
        """
        eigenvalues = np.sort(eigenvalues)
        
        # Empirical CDF
        n = len(eigenvalues)
        empirical_cdf = np.arange(1, n + 1) / n
        
        # Theoretical MP CDF
        theoretical_cdf = self.cdf(eigenvalues)
        
        # KS statistic: max absolute difference
        ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))
        
        # Compute p-value using scipy
        # Note: This is approximate, true distribution requires corrections
        p_value = stats.kstest(eigenvalues, self.cdf)[1]
        
        return ks_stat, p_value
    
    def tail_probability(self, lambda_val: float) -> float:
        """
        Compute tail probability P(λ > lambda_val).
        
        Important for detecting anomalies in the tail.
        
        Args:
            lambda_val: Threshold value
            
        Returns:
            Tail probability
        """
        if lambda_val >= self.lambda_plus:
            return 0.0
        elif lambda_val <= self.lambda_minus:
            return 1.0
        else:
            return 1.0 - self.cdf(np.array([lambda_val]))[0]
    
    def __repr__(self) -> str:
        return (f"MarchenkoPastur(γ={self.gamma:.3f}, σ²={self.sigma_sq:.3f}, "
                f"support=[{self.lambda_minus:.3f}, {self.lambda_plus:.3f}])")


def compute_spectral_density(eigenvalues: np.ndarray, 
                             n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical spectral density from eigenvalues.
    
    Args:
        eigenvalues: Array of eigenvalues
        n_bins: Number of histogram bins
        
    Returns:
        (bin_centers, density_values)
    """
    hist, bin_edges = np.histogram(eigenvalues, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist


def detect_tail_anomalies(eigenvalues: np.ndarray,
                          mp_law: MarchenkoPasturLaw,
                          threshold: float = 0.01) -> Tuple[np.ndarray, int]:
    """
    Detect eigenvalues in the tail that shouldn't be there.
    
    Byzantine gradients often create outlier eigenvalues beyond MP support.
    
    Args:
        eigenvalues: Observed eigenvalues
        mp_law: Theoretical MP law for honest gradients
        threshold: Tail probability threshold for detection
        
    Returns:
        (anomalous_eigenvalues, num_anomalies)
    """
    lambda_plus = mp_law.lambda_plus
    
    # Find eigenvalues significantly beyond support
    # Allow small numerical tolerance
    tolerance = 0.05 * lambda_plus
    anomalous = eigenvalues[eigenvalues > lambda_plus + tolerance]
    
    return anomalous, len(anomalous)
