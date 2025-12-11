import torch
import numpy as np
from src.utils.dataset import MNISTSampler
from src.utils.flow import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from architecture.nn_models import IsingNet
from src.utils.trainer import CFGTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data = MNISTSampler(),
    p_simple_shape = [1, 32, 32],
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
trainer = CFGTrainer(path = path, model = unet, eta=0.1)

# Train!
loss_history = trainer.train(num_epochs = 5000, device=device, lr=1e-3, batch_size=250)

# Save final weights for later reuse
torch.save(unet.state_dict(), "final_model.pth")

# Save training loss history
np.save("loss_history.npy", np.array(loss_history))

