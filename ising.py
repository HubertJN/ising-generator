import torch
from src.utils.dataset import IsingSampler
from src.utils.flow import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from architecture.nn_models import IsingNet
from src.utils.trainer import IsingTrainer
import os
import numpy as np

os.environ['CUDA_ALLOC_CONF'] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize our sampler
h5path = "../../data/gridstates_0.588_0.100.hdf5"
sampler = IsingSampler(h5path)

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data = sampler,
    p_simple_shape = [1, 64, 64],
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

# Initialize model
unet = IsingNet(
    channels = [32, 64, 128],
    num_residual_layers = 2,
    t_embed_dim = 40,
    y_embed_dim = 40,
)

# Initialize trainer
trainer = IsingTrainer(path = path, model = unet, eta=0.1)

# Train!
loss_history = trainer.train(num_epochs = 5000, device=device, lr=1e-3, batch_size=128)

# Save final weights for later reuse
torch.save(unet.state_dict(), "ising_model.pth")

# Save training loss history
np.save("ising_loss.npy", np.array(loss_history))