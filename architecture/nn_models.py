from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1) # (bs, 1)
        freqs = t * self.weights * 2 * math.pi # (bs, half_dim)
        sin_embed = torch.sin(freqs) # (bs, half_dim)
        cos_embed = torch.cos(freqs) # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2) # (bs, dim)
    
class ResidualLayer(nn.Module):
    def __init__(self, channels: int, time_dim: int, y_dim: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='circular')
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='circular')
        )
        self.time_adapter = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, channels)
        )
        self.y_adapter = nn.Sequential(
            nn.Linear(y_dim, y_dim),
            nn.SiLU(),
            nn.Linear(y_dim, channels)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        res = x.clone()

        x = self.block1(x)

        t_emb = self.time_adapter(t).unsqueeze(-1).unsqueeze(-1)
        x = x + t_emb

        y_emb = self.y_adapter(y).unsqueeze(-1).unsqueeze(-1)
        x = x + y_emb

        x = self.block2(x)

        return x + res

class Encoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, time_dim: int, y_dim: int, num_res_layers: int):
        super().__init__()
        self.res_layers = nn.ModuleList([
            ResidualLayer(channels_in, time_dim, y_dim) for _ in range(num_res_layers)
        ])
        self.downsample = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, padding_mode='circular')

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for layer in self.res_layers:
            x = layer(x, t, y)
        x = self.downsample(x)
        return x

class Decoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, time_dim: int, y_dim: int, num_res_layers: int):
        super().__init__()
        self.res_layers = nn.ModuleList([
            ResidualLayer(channels_out, time_dim, y_dim) for _ in range(num_res_layers)
        ])
        self.upsample = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        for layer in self.res_layers:
            x = layer(x, t, y)
        return x
    
class Midcoder(nn.Module):
    def __init__(self, channels: int, time_dim: int, y_dim: int, num_res_layers: int):
        super().__init__()
        self.res_layers = nn.ModuleList([
            ResidualLayer(channels, time_dim, y_dim) for _ in range(num_res_layers)
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for layer in self.res_layers:
            x = layer(x, t, y)
        return x

class IsingNet(nn.Module):
    def __init__(self, channels: List[int], time_dim: int, y_dim: int, num_res_layers: int):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU()
        )

        self.time_embedder = FourierEncoder(time_dim)

        self.y_embedder = nn.Embedding(num_embeddings = 11, embedding_dim = y_dim)

        encoders = []
        decoders = []
        for (curr_ch, next_ch) in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_ch, next_ch, time_dim, y_dim, num_res_layers))
            decoders.append(Decoder(next_ch, curr_ch, time_dim, y_dim, num_res_layers))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders[::-1])
    
        self.midcoder = Midcoder(channels[-1], time_dim, y_dim, num_res_layers)

        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1, padding_mode='circular')

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedder(t)
        y_emb = self.y_embedder(y)

        x = self.init_conv(x)

        residuals = []

        for encoder in self.encoders:
            x = encoder(x, t_emb, y_emb)
            residuals.append(x.clone())

        x = self.midcoder(x, t_emb, y_emb)

        for decoder in self.decoders:
            res = residuals.pop()
            x = x + res
            x = decoder(x, t_emb, y_emb)

        x = self.final_conv(x)

        return x
    
class TinyField(nn.Module):
    def __init__(self, L, time_dim=64):
        super().__init__()
        self.L = L
        self.time_mlp = FourierEncoder(time_dim)
        # simple 1x1 conv: elementwise affine map
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

        # time -> scale, bias
        self.t_to_ab = nn.Sequential(
            nn.Linear(time_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2)  # outputs [a(t), b(t)]
        )

    def forward(self, x, t):
        B, C, H, W = x.shape
        t_emb = self.time_mlp(t)    # (B, time_dim)
        ab = self.t_to_ab(t_emb)    # (B, 2)
        a = ab[:, 0].view(B, 1, 1, 1)
        b = ab[:, 1].view(B, 1, 1, 1)

        # elementwise affine in x
        y = a * x + b
        return self.conv(y)
