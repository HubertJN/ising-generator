from .dataset import IsingSampler, GaussianBaseSampler
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

class LinearFlow:
    def __init__(self, p_data: IsingSampler, p_base: GaussianBaseSampler):
        self.p_data = p_data
        self.p_base = p_base

    def sample_conditional_path(self, z, t):
        x0 = self.p_base.sample(z.shape[0]).to(z.device)  
        return (1.0-t)*x0 + t*z
             
    def conditional_vector_field(self, x, z, t):
        return (z - x) / (1.0 - t + 1e-8)
    
class GaussianFlow:
    def __init__(self, p_data: IsingSampler, p_base: GaussianBaseSampler):
        self.p_data = p_data
        self.p_base = p_base

    def sample_conditional_path(self, z, t):
        x0 = self.p_base.sample(z.shape[0]).to(z.device)  
        std = torch.sqrt(t)
        mean = (1 - t) * x0 + t * z
        eps = torch.randn_like(x0)
        return mean + std * eps
             
    def conditional_vector_field(self, x, z, t):
        std = torch.sqrt(t)
        mean = (1 - t) * x + t * z
        return (mean - x) / (std + 1e-8)

class FlowSimulator:
    def __init__(self, model):
        self.model = model

    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs):
        """
        Takes one Euler simulation step
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
            - dt: time step, shape (bs, 1, 1, 1)
        Returns:
            - nxt: state at time t + dt (bs, c, h, w)
        """
        v = self.model(xt, t)
        return xt + v * dt

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization given by ts
        Args:
            - x: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
        Returns:
            - x_final: final state at time ts[-1], shape (bs, c, h, w)
        """
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization given by ts
        Args:
            - x: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, nts, c, h, w)
        """
        xs = [x.clone()]
        nts = ts.shape[1]

        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)