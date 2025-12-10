import torch
import torch.nn as nn
import torch.optim
from tqdm import tqdm
from .flow import LinearFlow
import torch.nn.functional as F

class IsingTrainer:
    def __init__(self, path: LinearFlow, model: nn.Module):
        self.path = path
        self.model = model

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z, _ = self.path.p_data.sample(batch_size)
                
        t = torch.rand(batch_size,1,1,1).to(z)
        x = self.path.sample_conditional_path(z,t)

        ut_theta = self.model(x,t)
        ut_ref = self.path.conditional_vector_field(x,z,t)
        error = torch.einsum('bchw -> b', torch.square(ut_theta - ut_ref))
        return torch.mean(error)
    
    def train(self, num_epochs: int, device: str, lr: float, batch_size: int):
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            optimizer.zero_grad()
            loss = self.get_train_loss(batch_size)
            loss.backward()
            optimizer.step()
            pbar.set_description(f'Epoch {epoch}, loss: {loss.item()}')