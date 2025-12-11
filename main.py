import torch
import torch.nn as nn
from src.utils.flow import GaussianFlow, CFGSimulator
from src.utils.trainer import CFGTrainer
from architecture.nn_models import IsingNet
from src.utils.dataset import GaussianBaseSampler, MNISTSampler
import os

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Create samplers
sampler = MNISTSampler(device=device)
base = GaussianBaseSampler(channels=1, height=32, width=32)

# Create flow and model
flow = GaussianFlow(sampler, base)
model = IsingNet(
    channels=[32, 64, 128],
    time_dim=32,
    y_dim=32,
    num_res_layers=2
).to(device)

# Create trainer
trainer = CFGTrainer(flow, model, eta=0.1)

# Train the model
print("Starting training...")
loss_history = trainer.train(num_epochs=5000, device=device, lr=1e-3, batch_size=248)
print("Training completed.")

# Save the final model
save_model(model, 'final_model.pth')

