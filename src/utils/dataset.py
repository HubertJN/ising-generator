import torch
import torch.nn as nn
from typing import Optional

import sys
from pathlib import Path

parent_dir = Path.cwd().parent  # Adjust path as needed
sys.path.insert(0, str(parent_dir))

from modules.dataset import load_hdf5_raw, to_cnn_dataset

class IsingSampler:
    """
    Sampleable wrapper for the Ising dataset.
    """

    def __init__(self, hdf5_path: str, augment: bool = False, device: str = "cpu"):

        grids, attrs, _ = load_hdf5_raw(hdf5_path, load_grids=True)
        self.dataset = to_cnn_dataset(grids, attrs, augment=augment)
        self.device = device
        self.N = len(self.dataset)

    def sample(self, num_samples: int, device: Optional[str] = None):
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size, label_dim)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")
        
        if device is None:
            device = self.device

        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(device)
        labels = torch.stack(labels).to(device)
        return samples, labels

class GaussianBaseSampler:
    """
    Base distribution sampler for flow matching.
    Samples z ~ N(0, I) with the same spatial shape as the Ising configs.
    """

    def __init__(self, channels: int, height: int, width: int):
        """
        Parameters
        ----------
        channels : int
            Number of channels C (e.g. 1).
        height : int
            Spatial height H.
        width : int
            Spatial width W.
        """
        super().__init__()
        self.C = channels
        self.H = height
        self.W = width

    def sample(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Args:
            num_samples: the desired number of samples

        Returns:
            samples: shape (num_samples, C, H, W), z ~ N(0, I)
        """
        return torch.randn(num_samples, self.C, self.H, self.W, device=device)
    
class GlobalFlipSampler(nn.Module):
    def __init__(self, L: int, device: str = "cpu"):
        super().__init__()
        self.L = L
        self.device = device
        base = torch.ones(1, 1, L, L)
        self.register_buffer("base", base)

    def sample(self, batch_size: int, device: str | None = None):
        if device is None:
            device = self.device
        base = self.base.expand(batch_size, 1, self.L, self.L).to(device)
        flips = torch.randint(0, 2, (batch_size, 1, 1, 1), device=device)
        flips = 2 * flips - 1  # 0->-1, 1->+1
        z = base * flips
        return z, None
    
class GlobalUpSampler(nn.Module):
    def __init__(self, L: int, device: str = "cpu"):
        super().__init__()
        self.L = L
        self.device = device
        base = torch.ones(1, 1, L, L)
        self.register_buffer("base", base)

    def sample(self, batch_size: int, device: str | None = None):
        if device is None:
            device = self.device
        base = self.base.expand(batch_size, 1, self.L, self.L).to(device)
        return base, None
