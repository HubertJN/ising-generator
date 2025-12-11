import torch
import torch.nn as nn
import torch.optim
from tqdm import tqdm
import torch.nn.functional as F

def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size

MiB = 1024 * 1024

class CFGTrainer:
    def __init__(self, path, model, eta: float):
        assert eta > 0 and eta < 1
        super().__init__()
        self.model = model
        self.eta = eta
        self.path = path

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
    
    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> list[float]:
        # Report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / MiB:.3f} MiB')
        
        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(
            enumerate(range(num_epochs)),
            total=num_epochs,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        loss_history: list[float] = []
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item():.3f}')
            loss_history.append(loss.item())

        # Finish
        self.model.eval()
        return loss_history