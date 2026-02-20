"""
Oversmoothing metrics for tracking token representation collapse during training.

Metrics tracked:
- Effective rank of token embedding matrix (drops toward 1 = oversmoothing)
- Average pairwise cosine similarity (increases toward 1 = oversmoothing)
- Attention entropy (very high = uniform attention, very low = too peaked)
- High-frequency energy ratio (drops toward 0 = loss of fine-grained variation)
"""

import time

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange

def compute_effective_rank(embeddings: torch.Tensor) -> float:
    """Compute effective rank of token embedding matrix using singular values."""
    if embeddings.dim() == 3:
        embeddings = embeddings.mean(dim=0)  # [T, D]
    
    _, S, _ = torch.linalg.svd(embeddings, full_matrices=False)
    S_normalized = S / S.sum()
    S_normalized = S_normalized + 1e-10
    
    entropy = -torch.sum(S_normalized * torch.log(S_normalized))
    
    effective_rank = torch.exp(entropy)
    return effective_rank.item()


def compute_effective_rank_per_example(embeddings: torch.Tensor, centered: bool = False) -> float:
    """
    Compute effective rank of each EXAMPLE's token embedding matrix using singular values,
    then averages across examples.
    embeddings: [B, T, D]
    """
    if centered:
        embeddings = embeddings - embeddings.mean(dim=1, keepdim=True)  # Center per example

    # start = time.time()
    # _, S, _ = torch.linalg.svd(embeddings, full_matrices=False)
    # print("SVD time", time.time() - start)
    # start = time.time()
    S = torch.linalg.svdvals(embeddings)
    # print("SVDvals time", time.time() - start)
    # assert torch.allclose(sv, S), "SVD values mismatch"
    # print("S shapes", S.shape, sv.shape)

    S_normalized = S / S.sum(dim=1, keepdim=True)  # S should be [B, rank]. Normalize per example
    S_normalized = S_normalized + 1e-10

    entropy = -torch.sum(S_normalized * torch.log(S_normalized), dim=1)  # Compute entropy per example

    effective_rank = torch.exp(entropy).mean()  # Average across examples
    return effective_rank.item()


def compute_same_diff_distance_ratio(embeddings: torch.Tensor) -> float:
    """
    For each example, compute centroid of its token embeddings.
    Then compute average distance of tokens to their own centroid (intra) vs distance between centroids (inter).
    Lower ratio = tokens from same example are clustered, suggesting oversmoothing.

    Args:
        embeddings: [B, T, D] token embeddings
    Returns:
        intra-to-inter distance ratio (scalar)
    """
    B, T, D = embeddings.shape
    centroids = embeddings.mean(dim=1, keepdim=True)  # [B, 1, D]
    dist_to_centroid = torch.norm(embeddings - centroids, dim=-1)  # [B, T]
    dist_between_centroid = F.pdist(centroids.squeeze(1))  # [B*(B-1)/2]
    return dist_to_centroid.mean().item() / (dist_between_centroid.mean().item() + 1e-10)


def compute_avg_cosine_similarity(embeddings: torch.Tensor) -> float:
    """
    Compute average pairwise cosine similarity between token embeddings.
    Higher values indicate more similar tokens (potential oversmoothing).
    
    Args:
        embeddings: [B, T, D] token embeddings
    Returns:
        average cosine similarity (scalar)
    """
    embeddings_norm = F.normalize(embeddings, p=2, dim=-1)  # [B, T, D]
    
    # Compute pairwise cosine similarity: [B, T, T]
    similarity_matrix = torch.bmm(embeddings_norm, embeddings_norm.transpose(1, 2))    
    _, T, _ = similarity_matrix.shape
    mask = torch.triu(torch.ones(T, T, device=embeddings.device), diagonal=1).bool()
    
    # Extract upper triangular values and compute mean
    upper_tri_values = similarity_matrix[:, mask]  # [B, T*(T-1)/2]
    avg_similarity = upper_tri_values.mean().item()
    
    return avg_similarity


def compute_attention_effective_rank(attn_weights: torch.Tensor) -> float:
    """
    Compute effective rank of each attention matrix, and average across examples/heads.
    Returns average per layer.
    Very low effective rank (close to 1) indicates each token's attention focuses on the same tokens.

    Args:
        attn_weights: [L, B, H, T, T] attention weights (post-softmax)
    Returns:
        effective rank per layer: [L] array
    """
    assert attn_weights.dim() == 5, "Expected attn_weights to be [L, B, H, T, T]"
    L, B, H, T, _ = attn_weights.shape
    effective_ranks = []
    for layer_idx in range(L):
        layer_attn = attn_weights[layer_idx]  # [B, H, T, T]
        attn_matrices = rearrange(layer_attn, 'b h t1 t2 -> (b h) t1 t2')  # [B*H, T, T]

        S = torch.linalg.svdvals(attn_matrices)  # [B*H, min(T, T)]
        sv_normalized = S / S.sum(dim=1, keepdim=True)  # Normalize per example
        sv_normalized = sv_normalized + 1e-10
        entropy = -torch.sum(sv_normalized * torch.log(sv_normalized), dim=1)  # [B*H]
        eff_rank = torch.exp(entropy).mean().item()  # Average effective rank across examples/heads
        effective_ranks.append(eff_rank)
    return np.array(effective_ranks)


def compute_attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of attention distributions. Low entropy = peaked attention,
    High entropy = uniform attention (potential oversmoothing in attention).
    
    Args:
        attn_weights: [L, B, H, T, T] or [B, H, T, T] attention weights (post-softmax)
    Returns:
        entropy per layer: [L] tensor or scalar tensor
    """
    attn_weights = attn_weights + 1e-10
    
    # Entropy along the last dimension
    entropy = -torch.sum(attn_weights * torch.log(attn_weights), dim=-1) 
    
    if attn_weights.dim() == 5:  # [L, B, H, T, T]
        entropy_per_layer = entropy.mean(dim=(1, 2, 3))  # [L]
    else:  # [B, H, T, T]
        entropy_per_layer = entropy.mean()
    
    return entropy_per_layer


def compute_attention_entropy_per_head(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of attention distributions per head.
    
    Args:
        attn_weights: [L, B, H, T, T] or [B, H, T, T] attention weights (post-softmax)
    Returns:
        entropy per layer per head: [L, H] or [H] tensor
    """
    attn_weights = attn_weights + 1e-10
    
    # Entropy along the last dimension (attention distribution)
    entropy = -torch.sum(attn_weights * torch.log(attn_weights), dim=-1)
    
    if attn_weights.dim() == 5:  # [L, B, H, T, T] -> entropy is [L, B, H, T]
        # Average over batch and queries, keep layer and head dimensions
        # [L, B, H, T] -> [L, H]
        entropy_per_layer_head = entropy.mean(dim=(1, 3))
    elif attn_weights.dim() == 4:  # [B, H, T, T] -> entropy is [B, H, T]
        # Average over batch and queries, keep head dimension
        # [B, H, T] -> [H]
        entropy_per_layer_head = entropy.mean(dim=(0, 2))
    else:
        entropy_per_layer_head = entropy.mean()
    
    return entropy_per_layer_head


def compute_high_freq_energy_ratio(embeddings: torch.Tensor, cutoff_ratio: float = 0.5) -> float:
    """
    For each channel, treat the sequence of token values as a 1D signal and compute 
    its Fourier spectrum. Track the ratio of high-frequency energy to total energy.
    
    Lower ratio = more low-frequency content = smoother signals (potential oversmoothing)
    
    Args:
        embeddings: [B, T, D] token embeddings
        cutoff_ratio: fraction of frequencies considered "high frequency" (default 0.5 = upper half)
    Returns:
        ratio of high-frequency energy to total energy
    """
    # Compute FFT along the time dimension for each channel
    # Result: [B, T//2+1, D] complex tensor
    fft_result = torch.fft.rfft(embeddings, dim=1)    
    power_spectrum = torch.abs(fft_result) ** 2  # [B, T//2+1, D]
    
    n_freqs = power_spectrum.shape[1]
    cutoff_idx = int(n_freqs * (1 - cutoff_ratio)) 
    
    # Sum energy in high frequencies vs total
    total_energy = power_spectrum.sum()
    high_freq_energy = power_spectrum[:, cutoff_idx:, :].sum()
    
    ratio = (high_freq_energy / (total_energy + 1e-10)).item()
    return ratio


class OverSmoothingMetrics:
    """Helper class to collect and track oversmoothing metrics during training.
    
    Metrics tracked:
    - Effective rank of token embedding matrix (drops toward 1 = oversmoothing)
    - Average pairwise cosine similarity (increases toward 1 = oversmoothing)
    - Attention entropy (very high = uniform attention, very low = too peaked)
    - High-frequency energy ratio (drops toward 0 = loss of fine-grained variation)
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics for a new epoch."""
        self.effective_ranks = []
        self.cosine_similarities = []  
        self.attention_entropies = [] 
        self.attention_entropies_per_head = []
        self.high_freq_ratios = []
        self.effective_ranks_per_example = []
        self.same_diff_ratios = []
        self.attention_effective_ranks = []

    
    def compute_metrics(self, embeddings_layers: torch.Tensor, attn_weights: torch.Tensor, max_examples: int = None) -> dict:
        """
        Compute all oversmoothing metrics for a single batch.
        
        Args:
            embeddings_layers: [L+1, B, T, D] - embeddings at start and after each layer
            attn_weights: [L, B, H, T, T] - attention weights for each layer
        
        Returns:
            dict with metrics per layer
        """
        num_layers = embeddings_layers.shape[0]  # L+1 (including input)
        if max_examples is not None:
            embeddings_layers = embeddings_layers[:, :max_examples]  # Limit to max examples for faster computation
            attn_weights = attn_weights[:, :max_examples]
        
        # Per-layer metrics for embeddings
        layer_effective_ranks = []
        layer_cosine_sims = []
        layer_high_freq_ratios = []
        layer_effective_ranks_per_example = []
        layer_same_diff_ratios = []

        for layer_idx in range(num_layers):
            embed = embeddings_layers[layer_idx]  # [B, T, D]
            
            eff_rank = compute_effective_rank(embed)
            layer_effective_ranks.append(eff_rank)
            
            cos_sim = compute_avg_cosine_similarity(embed)
            layer_cosine_sims.append(cos_sim)
            
            hf_ratio = compute_high_freq_energy_ratio(embed)
            layer_high_freq_ratios.append(hf_ratio)

            eff_rank_per_example = compute_effective_rank_per_example(embed)
            layer_effective_ranks_per_example.append(eff_rank_per_example)

            same_diff_ratio = compute_same_diff_distance_ratio(embed)
            layer_same_diff_ratios.append(same_diff_ratio)
        
        attn_entropy = compute_attention_entropy(attn_weights)
        attn_entropy_per_head = compute_attention_entropy_per_head(attn_weights)
        attn_effective_rank = compute_attention_effective_rank(attn_weights)

        return {
            'effective_rank': layer_effective_ranks, 
            'cosine_similarity': layer_cosine_sims,
            'attention_entropy': attn_entropy.detach().cpu().numpy(),
            'attention_entropy_per_head': attn_entropy_per_head.detach().cpu().numpy(),  # [L, H]
            'high_freq_ratio': layer_high_freq_ratios,
            'effective_rank_per_example': layer_effective_ranks_per_example,
            'same_diff_ratio': layer_same_diff_ratios,
            'attention_effective_rank': attn_effective_rank  # [L]
        }
    
    def accumulate(self, metrics_dict: dict):
        """Add metrics from a batch to running totals."""
        self.effective_ranks.append(metrics_dict['effective_rank'])
        self.cosine_similarities.append(metrics_dict['cosine_similarity'])
        self.attention_entropies.append(metrics_dict['attention_entropy'])
        self.attention_entropies_per_head.append(metrics_dict['attention_entropy_per_head'])
        self.high_freq_ratios.append(metrics_dict['high_freq_ratio'])
        self.effective_ranks_per_example.append(metrics_dict['effective_rank_per_example'])
        self.same_diff_ratios.append(metrics_dict['same_diff_ratio'])
        self.attention_effective_ranks.append(metrics_dict['attention_effective_rank'])

    
    def get_epoch_summary(self) -> dict:
        """
        Compute epoch-level averages.
        
        Returns:
            dict with averaged metrics:
            - 'effective_rank': [L+1] array, effective rank per layer
            - 'cosine_similarity': [L+1] array, avg cosine sim per layer
            - 'attention_entropy': [L] array, attention entropy per layer
            - 'attention_entropy_per_head': [L, H] array, attention entropy per layer per head
            - 'high_freq_ratio': [L+1] array, high-freq energy ratio per layer
            - 'effective_rank_per_example': [L+1] array, effective rank per example averaged across examples
            - 'same_diff_ratio': [L+1] array, intra-to-inter distance ratio per layer
            - 'attention_effective_rank': [L] array, effective rank of attention matrices per layer
        """
        if len(self.effective_ranks) == 0:
            # Maybe should just return None for simplicity? Not sure
            return {
                'effective_rank': np.array([]),
                'cosine_similarity': np.array([]),
                'attention_entropy': np.array([]),
                'attention_entropy_per_head': np.array([]),
                'high_freq_ratio': np.array([]),
                'effective_rank_per_example': np.array([]),
                'same_diff_ratio': np.array([]),
                'attention_effective_rank': np.array([])
            }
        
        return {
            'effective_rank': np.atleast_1d(np.mean(self.effective_ranks, axis=0)),
            'cosine_similarity': np.atleast_1d(np.mean(self.cosine_similarities, axis=0)),
            'attention_entropy': np.atleast_1d(np.mean(self.attention_entropies, axis=0)),
            'attention_entropy_per_head': np.mean(self.attention_entropies_per_head, axis=0),  # [L, H]
            'high_freq_ratio': np.atleast_1d(np.mean(self.high_freq_ratios, axis=0)),
            'effective_rank_per_example': np.atleast_1d(np.mean(self.effective_ranks_per_example, axis=0)),
            'same_diff_ratio': np.atleast_1d(np.mean(self.same_diff_ratios, axis=0)),
            'attention_effective_rank': np.atleast_1d(np.mean(self.attention_effective_ranks, axis=0))
        }
