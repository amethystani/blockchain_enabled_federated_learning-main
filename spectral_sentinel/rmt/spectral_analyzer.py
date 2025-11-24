"""
Spectral Analyzer for Byzantine Detection

Uses Random Matrix Theory to analyze gradient matrices and detect
Byzantine adversaries via spectral anomalies.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from collections import deque
import warnings

from spectral_sentinel.rmt.marchenko_pastur import (
    MarchenkoPasturLaw, 
    compute_spectral_density,
    detect_tail_anomalies
)


class SpectralAnalyzer:
    """
    Analyze gradient matrices using RMT to detect Byzantine clients.
    
    Core idea: Honest gradients (even with Non-IID data) follow Marchenko-Pastur
    law in their eigenspectrum. Byzantine gradients create detectable anomalies.
    """
    
    def __init__(self,
                 ks_threshold: float = 0.05,
                 tail_threshold: float = 0.1,
                 window_size: int = 50,
                 adaptive: bool = True):
        """
        Initialize spectral analyzer.
        
        Args:
            ks_threshold: P-value threshold for KS test (reject if p < threshold)
            tail_threshold: Fraction of eigenvalues allowed in tail
            window_size: Sliding window for online MP tracking
            adaptive: Use adaptive threshold calibration
        """
        self.ks_threshold = ks_threshold
        self.tail_threshold = tail_threshold
        self.window_size = window_size
        self.adaptive = adaptive
        
        # Online tracking
        self.history: deque = deque(maxlen=window_size)
        self.mp_law: Optional[MarchenkoPasturLaw] = None
        
        # Statistics
        self.stats = {
            'ks_statistics': [],
            'p_values': [],
            'tail_anomalies': [],
            'detected_byzantine': [],
            'phase_transition_metric': [],  # σ²f² values
            'heterogeneity_sigma': [],  # Coordinate-wise variance
            'convergence_rate': []  # Actual convergence tracking
        }
    
    def analyze_gradients(self, 
                         gradients: List[torch.Tensor],
                         client_ids: Optional[List[int]] = None) -> Dict:
        """
        Analyze gradients from multiple clients for Byzantine detection.
        
        Args:
            gradients: List of gradient tensors, one per client
            client_ids: Optional client IDs for tracking
            
        Returns:
            Detection results dictionary
        """
        if client_ids is None:
            client_ids = list(range(len(gradients)))
        
        # Convert gradients to matrix form: n_clients × d_params
        gradient_matrix = self._gradients_to_matrix(gradients)
        n_clients, d_params = gradient_matrix.shape
        
        # Compute covariance matrix eigenvalues
        # C = (1/n) G^T G where G is n×d gradient matrix
        eigenvalues = self._compute_eigenvalues(gradient_matrix)
        
        # Fit or update MP law
        if self.mp_law is None or self.adaptive:
            self.mp_law = self._fit_mp_law(eigenvalues, n_clients, d_params)
        
        # Detect anomalies
        results = self._detect_anomalies(eigenvalues, client_ids, gradient_matrix)
        
        # Calculate phase transition metric (σ²f²)
        phase_metric = self._calculate_phase_transition_metric(
            gradient_matrix, results
        )
        results['phase_transition_metric'] = phase_metric
        
        # Update history
        self.history.append({
            'eigenvalues': eigenvalues,
            'mp_law': self.mp_law,
            'results': results
        })
        
        return results
    
    def _gradients_to_matrix(self, gradients: List[torch.Tensor]) -> np.ndarray:
        """
        Convert list of gradient tensors to matrix.
        
        Args:
            gradients: List of gradient tensors (possibly nested)
            
        Returns:
            Matrix of shape (n_clients, d_params)
        """
        # Flatten each gradient tensor
        flattened = []
        for g in gradients:
            if isinstance(g, dict):
                # Handle dict of gradients (layer-wise)
                flat = torch.cat([param.flatten() for param in g.values()])
            elif isinstance(g, list):
                # Handle list of tensors
                flat = torch.cat([param.flatten() for param in g])
            else:
                # Single tensor
                flat = g.flatten()
            
            flattened.append(flat.detach().cpu().numpy())
        
        return np.vstack(flattened)
    
    def _compute_eigenvalues(self, gradient_matrix: np.ndarray) -> np.ndarray:
        """
        Compute eigenvalues of empirical covariance matrix.
        
        Args:
            gradient_matrix: n × d matrix
            
        Returns:
            Sorted eigenvalues (descending)
        """
        n, d = gradient_matrix.shape
        
        # Compute sample covariance: (1/n) G^T G
        # For computational efficiency, use SVD if d >> n
        if d > 10 * n:
            # Use SVD: G = UΣV^T, then G^TG has eigenvalues σ²/n
            _, singular_values, _ = np.linalg.svd(gradient_matrix, full_matrices=False)
            eigenvalues = (singular_values ** 2) / n
        else:
            # Direct eigenvalue decomposition
            cov_matrix = (gradient_matrix.T @ gradient_matrix) / n
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        # Sort descending
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        return eigenvalues
    
    def _fit_mp_law(self, 
                    eigenvalues: np.ndarray,
                    n_samples: int,
                    n_features: int) -> MarchenkoPasturLaw:
        """
        Fit Marchenko-Pastur law to observed eigenvalues.
        
        Args:
            eigenvalues: Observed eigenvalues
            n_samples: Number of clients
            n_features: Number of parameters
            
        Returns:
            Fitted MP law
        """
        aspect_ratio = n_samples / n_features
        
        # Estimate variance from eigenvalues
        # Remove potential outliers for robust estimation
        clean_eigenvalues = self._robust_eigenvalue_filter(eigenvalues)
        
        mean_lambda = np.mean(clean_eigenvalues)
        sigma_sq = mean_lambda / (1 + 1/aspect_ratio)
        
        return MarchenkoPasturLaw(aspect_ratio, sigma_sq)
    
    def _robust_eigenvalue_filter(self, eigenvalues: np.ndarray,
                                  quantile: float = 0.95) -> np.ndarray:
        """
        Remove outlier eigenvalues for robust MP fitting.
        
        Args:
            eigenvalues: All eigenvalues
            quantile: Keep eigenvalues below this quantile
            
        Returns:
            Filtered eigenvalues
        """
        threshold = np.quantile(eigenvalues, quantile)
        return eigenvalues[eigenvalues <= threshold]
    
    def _detect_anomalies(self,
                         eigenvalues: np.ndarray,
                         client_ids: List[int],
                         gradient_matrix: np.ndarray) -> Dict:
        """
        Detect Byzantine clients via spectral anomalies.
        
        Args:
            eigenvalues: Eigenvalues of gradient covariance
            client_ids: Client identifiers
            gradient_matrix: n × d gradient matrix
            
        Returns:
            Detection results
        """
        results = {
            'byzantine_detected': [],
            'honest_clients': [],
            'ks_statistic': 0.0,
            'ks_pvalue': 1.0,
            'tail_anomalies': 0,
            'detection_method': None
        }
        
        # Test 1: Kolmogorov-Smirnov test for MP conformance
        ks_stat, p_value = self.mp_law.ks_test(eigenvalues)
        results['ks_statistic'] = ks_stat
        results['ks_pvalue'] = p_value
        
        self.stats['ks_statistics'].append(ks_stat)
        self.stats['p_values'].append(p_value)
        
        # Test 2: Tail anomaly detection
        anomalous_eigenvalues, n_anomalies = detect_tail_anomalies(
            eigenvalues, self.mp_law, self.tail_threshold
        )
        results['tail_anomalies'] = n_anomalies
        self.stats['tail_anomalies'].append(n_anomalies)
        
        # Decision logic: Detect if EITHER test fails
        if p_value < self.ks_threshold or n_anomalies > 0:
            # Byzantine activity detected - identify culprits
            byzantine_clients = self._identify_byzantine_clients(
                gradient_matrix, eigenvalues, client_ids
            )
            results['byzantine_detected'] = byzantine_clients
            results['honest_clients'] = [
                cid for cid in client_ids if cid not in byzantine_clients
            ]
            results['detection_method'] = 'spectral_anomaly'
        else:
            # All clients appear honest
            results['honest_clients'] = client_ids
            results['detection_method'] = 'no_detection'
        
        self.stats['detected_byzantine'].append(len(results['byzantine_detected']))
        
        return results
    
    def _identify_byzantine_clients(self,
                                    gradient_matrix: np.ndarray,
                                    eigenvalues: np.ndarray,
                                    client_ids: List[int]) -> List[int]:
        """
        Identify which specific clients are Byzantine.
        
        Strategy: Project gradients onto top eigenvectors (anomalous directions)
        and flag clients with large projections.
        
        Args:
            gradient_matrix: n × d gradient matrix
            eigenvalues: Eigenvalues
            client_ids: Client IDs
            
        Returns:
            List of Byzantine client IDs
        """
        n_clients = len(client_ids)
        
        # Find anomalous eigenvalues (beyond MP support)
        lambda_plus = self.mp_law.lambda_plus
        tolerance = 0.05 * lambda_plus
        anomalous_mask = eigenvalues > (lambda_plus + tolerance)
        
        if not np.any(anomalous_mask):
            # No anomalous eigenvalues, fall back to statistical outlier detection
            return self._statistical_outlier_detection(gradient_matrix, client_ids)
        
        # Compute eigenvectors corresponding to anomalous eigenvalues
        n, d = gradient_matrix.shape
        cov_matrix = (gradient_matrix.T @ gradient_matrix) / n
        eigenvalues_full, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort descending
        idx = np.argsort(eigenvalues_full)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Select anomalous eigenvectors
        anomalous_indices = np.where(anomalous_mask)[0]
        anomalous_eigenvectors = eigenvectors[:, anomalous_indices]
        
        # Project each client's gradient onto anomalous subspace
        projections = np.abs(gradient_matrix @ anomalous_eigenvectors)
        projection_norms = np.linalg.norm(projections, axis=1)
        
        # Flag clients with large projections (top 25% or above threshold)
        threshold = np.percentile(projection_norms, 75)
        byzantine_mask = projection_norms > threshold
        
        byzantine_clients = [client_ids[i] for i in range(n_clients) 
                            if byzantine_mask[i]]
        
        return byzantine_clients
    
    def _statistical_outlier_detection(self,
                                       gradient_matrix: np.ndarray,
                                       client_ids: List[int],
                                       z_threshold: float = 2.5) -> List[int]:
        """
        Fallback: Detect outliers using gradient norm statistics.
        
        Args:
            gradient_matrix: n × d gradient matrix
            client_ids: Client IDs
            z_threshold: Z-score threshold for outlier detection
            
        Returns:
            List of outlier client IDs
        """
        # Compute gradient norms
        norms = np.linalg.norm(gradient_matrix, axis=1)
        
        # Z-score based detection
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        if std_norm < 1e-8:
            return []
        
        z_scores = np.abs((norms - mean_norm) / std_norm)
        outlier_mask = z_scores > z_threshold
        
        outliers = [client_ids[i] for i in range(len(client_ids)) 
                   if outlier_mask[i]]
        
        return outliers
    
    def _calculate_phase_transition_metric(self,
                                           gradient_matrix: np.ndarray,
                                           detection_results: Dict) -> Dict:
        """
        Calculate phase transition metric σ²f².
        
        Critical threshold: σ²f² < 0.25 for reliable detection
        
        Args:
            gradient_matrix: n × d gradient matrix
            detection_results: Detection results with Byzantine counts
            
        Returns:
            Dict with phase transition info
        """
        n_clients, d_params = gradient_matrix.shape
        
        # Estimate coordinate-wise variance σ²
        # Use robust estimator (median absolute deviation)
        coord_variance = np.var(gradient_matrix, axis=0)
        sigma_sq = np.median(coord_variance)
        
        # Byzantine fraction f
        n_byzantine = len(detection_results['byzantine_detected'])
        f = n_byzantine / n_clients if n_clients > 0 else 0.0
        
        # Phase transition metric
        phase_metric = sigma_sq * (f ** 2)
        
        # Track in statistics
        self.stats['phase_transition_metric'].append(phase_metric)
        self.stats['heterogeneity_sigma'].append(np.sqrt(sigma_sq))
        
        # Warning system
        phase_status = "safe"
        warning_message = None
        
        if phase_metric >= 0.25:
            phase_status = "impossible"
            warning_message = (f"⚠️  CRITICAL: σ²f² = {phase_metric:.4f} ≥ 0.25 "
                             f"(Phase transition exceeded! Detection theoretically impossible)")
        elif phase_metric >= 0.20:
            phase_status = "near_transition"
            warning_message = (f"⚠️  WARNING: σ²f² = {phase_metric:.4f} approaching 0.25 "
                             f"(Near phase transition, detection degrading)")
        elif phase_metric >= 0.15:
            phase_status = "elevated"
            warning_message = f"⚡ CAUTION: σ²f² = {phase_metric:.4f} (Elevated, monitor closely)"
        
        if warning_message:
            print(warning_message)
        
        return {
            'sigma_squared_f_squared': phase_metric,
            'sigma': np.sqrt(sigma_sq),
            'f': f,
            'status': phase_status,
            'detectable': phase_metric < 0.25,
            'warning': warning_message
        }
    
    def calculate_convergence_rate(self, 
                                   round_num: int,
                                   total_rounds: int) -> Dict:
        """
        Calculate theoretical vs actual convergence rate.
        
        Theoretical: O(σf/√T + f²/T)
        
        Args:
            round_num: Current round
            total_rounds: Total rounds
            
        Returns:
            Convergence rate info
        """
        if len(self.stats['phase_transition_metric']) == 0:
            return {}
        
        # Get recent phase transition metric
        recent_phase = self.stats['phase_transition_metric'][-1]
        recent_sigma = self.stats['heterogeneity_sigma'][-1]
        
        # Extract f from recent detections
        if len(self.stats['detected_byzantine']) > 0:
            avg_f = np.mean([d for d in self.stats['detected_byzantine'][-10:]]) / 20  # Assume 20 clients
        else:
            avg_f = 0.0
        
        T = round_num
        
        # Theoretical convergence rate: O(σf/√T + f²/T)
        if T > 0:
            term1 = recent_sigma * avg_f / np.sqrt(T)
            term2 = (avg_f ** 2) / T
            theoretical_rate = term1 + term2
        else:
            theoretical_rate = float('inf')
        
        self.stats['convergence_rate'].append(theoretical_rate)
        
        return {
            'round': round_num,
            'theoretical_rate': theoretical_rate,
            'term1_sigma_f_sqrt_T': term1 if T > 0 else 0,
            'term2_f2_T': term2 if T > 0 else 0,
            'matches_optimal': recent_phase < 0.1  # σf = O(1) condition
        }
    
    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        return self.stats
    
    def reset_statistics(self):
        """Reset tracking statistics."""
        self.stats = {
            'ks_statistics': [],
            'p_values': [],
            'tail_anomalies': [],
            'detected_byzantine': []
        }
        self.history.clear()
    
    def visualize_spectrum(self, save_path: Optional[str] = None):
        """
        Visualize eigenvalue spectrum vs MP law.
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.history:
            warnings.warn("No data to visualize. Run analyze_gradients first.")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed, skipping visualization")
            return
        
        # Get most recent analysis
        latest = self.history[-1]
        eigenvalues = latest['eigenvalues']
        mp_law = latest['mp_law']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Eigenvalue spectrum
        ax1.hist(eigenvalues, bins=50, density=True, alpha=0.7, 
                label='Empirical', color='steelblue')
        
        # Overlay MP density
        x = np.linspace(mp_law.lambda_minus * 0.8, mp_law.lambda_plus * 1.2, 1000)
        y = mp_law.density(x)
        ax1.plot(x, y, 'r-', linewidth=2, label='MP Law')
        
        # Mark support bounds
        ax1.axvline(mp_law.lambda_minus, color='green', linestyle='--', 
                   label='MP Support')
        ax1.axvline(mp_law.lambda_plus, color='green', linestyle='--')
        
        ax1.set_xlabel('Eigenvalue λ')
        ax1.set_ylabel('Density')
        ax1.set_title('Spectral Density vs Marchenko-Pastur Law')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: QQ plot
        theoretical_quantiles = np.linspace(0.01, 0.99, len(eigenvalues))
        empirical_quantiles = np.sort(eigenvalues)
        
        # Compute theoretical quantiles from MP law (approximately)
        theoretical_eigenvalues = np.percentile(
            np.linspace(mp_law.lambda_minus, mp_law.lambda_plus, 10000),
            theoretical_quantiles * 100
        )
        
        ax2.scatter(theoretical_eigenvalues[:len(empirical_quantiles)], 
                   empirical_quantiles, alpha=0.6, s=20)
        
        # Add diagonal line
        min_val = min(empirical_quantiles.min(), theoretical_eigenvalues.min())
        max_val = max(empirical_quantiles.max(), theoretical_eigenvalues.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        ax2.set_xlabel('Theoretical Quantiles (MP Law)')
        ax2.set_ylabel('Empirical Quantiles')
        ax2.set_title('Q-Q Plot')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved spectrum visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
