"""
HL-Gauss (Histogram Loss with Gaussian Smoothing) for regression as classification.

Based on: "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL"
https://arxiv.org/abs/2403.03950

The key idea is to convert scalar regression targets into smoothed categorical distributions
and use cross-entropy loss instead of MSE. This provides:
- Better gradient properties (more stable)
- Improved representation learning
- Better handling of noisy/non-stationary targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HLGaussLoss(nn.Module):
    """Histogram Loss with Gaussian smoothing for regression as classification.
    
    Instead of regressing to scalar values directly, this loss:
    1. Discretizes the value range into bins
    2. Converts scalar targets to Gaussian-smoothed probability distributions over bins
    3. Uses cross-entropy loss between predicted and target distributions
    
    Args:
        min_value: Minimum value of the support range
        max_value: Maximum value of the support range
        num_bins: Number of bins in the histogram
        sigma_ratio: Ratio of sigma to bin width (default 0.75, distributes mass to ~6 bins)
    """
    
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma_ratio: float = 0.75):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        
        # Compute bin width and sigma
        self.bin_width = (max_value - min_value) / num_bins
        self.sigma = sigma_ratio * self.bin_width
        
        # Register support as buffer (bin edges)
        support = torch.linspace(min_value, max_value, num_bins + 1, dtype=torch.float32)
        self.register_buffer('support', support)
        
        # Precompute bin centers for recovering predictions
        bin_centers = (support[:-1] + support[1:]) / 2
        self.register_buffer('bin_centers', bin_centers)
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute HL-Gauss loss.
        
        Args:
            logits: Model output logits of shape [B, num_bins] or [B, 1, num_bins]
            target: Scalar targets of shape [B] or [B, 1]
            
        Returns:
            Scalar loss value (mean over batch)
        """
        if logits.dim() == 3:
            logits = logits.squeeze(1)  # [B, 1, num_bins] -> [B, num_bins]
        if target.dim() == 2:
            target = target.squeeze(1)  # [B, 1] -> [B]
            
        target_probs = self.transform_to_probs(target)  # [B, num_bins]
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(target_probs * log_probs, dim=-1)  # [B]
        
        return loss.mean()
    
    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        """Transform scalar targets to Gaussian-smoothed probability distributions.
        
        Args:
            target: Scalar targets of shape [B]
            
        Returns:
            Probability distribution over bins of shape [B, num_bins]
        """
        support = self.support.to(target.device)        
        target = torch.clamp(target, self.min_value + 1e-6, self.max_value - 1e-6)
        
        # erf((x - mu) / (sqrt(2) * sigma)) gives scaled CDF
        sqrt2 = math.sqrt(2.0)
        cdf_evals = torch.special.erf(
            (support - target.unsqueeze(-1)) / (sqrt2 * self.sigma)
        )  # [B, num_bins + 1]
        
        z = cdf_evals[..., -1] - cdf_evals[..., 0]  # [B]        
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]  # [B, num_bins]
        bin_probs = bin_probs / z.unsqueeze(-1)
        
        return bin_probs
    
    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Recover scalar predictions from probability distributions.
        
        Args:
            probs: Probability distribution over bins of shape [B, num_bins]
            
        Returns:
            Scalar predictions of shape [B]
        """
        bin_centers = self.bin_centers.to(probs.device)
        return torch.sum(probs * bin_centers, dim=-1)
    
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Get scalar predictions from model logits.
        
        Args:
            logits: Model output logits of shape [B, num_bins] or [B, 1, num_bins]
            
        Returns:
            Scalar predictions of shape [B]
        """
        if logits.dim() == 3:
            logits = logits.squeeze(1)
        
        probs = F.softmax(logits, dim=-1)
        return self.transform_from_probs(probs)


def compute_hl_gauss_range(labels: torch.Tensor, margin_factor: float = 1.2) -> tuple:
    """Compute reasonable min/max values for HL-Gauss from training labels.
    
    Args:
        labels: Training labels tensor
        margin_factor: Multiply range by this factor to add margin (default 1.2 = 20% margin)
        
    Returns:
        Tuple of (min_value, max_value)
    """
    label_min = labels.min().item()
    label_max = labels.max().item()
    
    label_range = label_max - label_min
    margin = label_range * (margin_factor - 1) / 2
    
    min_value = label_min - margin
    max_value = label_max + margin
    
    return min_value, max_value
