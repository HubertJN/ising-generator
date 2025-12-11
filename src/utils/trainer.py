import torch
import torch.nn as nn
import torch.optim
from tqdm import tqdm
from .flow import LinearFlow, GaussianFlow
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
        mse = torch.mean(torch.einsum('bchw -> b', torch.square(ut_theta - ut_ref)))
        error = mse
        return error
    
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

    def train_history(self, num_epochs: int, device: str, lr: float, batch_size: int):
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        loss_history = []
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            optimizer.zero_grad()
            loss = self.get_train_loss(batch_size)
            loss.backward()
            optimizer.step()
            pbar.set_description(f'Epoch {epoch}, loss: {loss.item()}')
            loss_history.append(loss.item())

        return loss_history
    

class CFGTrainer:
    def __init__(self, path: GaussianFlow, model: nn.Module, eta: float):
        assert eta > 0 and eta < 1
        super().__init__()
        self.eta = eta
        self.path = path
        self.model = model

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        z, y = self.path.p_data.sample(batch_size) # (bs, c, h, w), (bs,1)
        
        # Step 2: Set each label to 10 (i.e., null) with probability eta
        xi = torch.rand(y.shape[0]).to(y.device)
        y[xi < self.eta] = 10.0
        
        # Step 3: Sample t and x
        t = torch.rand(batch_size,1,1,1).to(z) # (bs, 1, 1, 1)
        x = self.path.sample_conditional_path(z,t) # (bs, 1, 32, 32)

        # Step 4: Regress and output loss
        ut_theta = self.model(x,t,y) # (bs, 1, 32, 32)
        ut_ref = self.path.conditional_vector_field(x,z,t) # (bs, 1, 32, 32)
        error = torch.einsum('bchw -> b', torch.square(ut_theta - ut_ref)) # (bs,)
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
