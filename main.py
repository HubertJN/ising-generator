import torch
import torch.nn as nn
from src.utils.flow import LinearFlow, FlowSimulator
from src.utils.trainer import IsingFlowTrainer
from architecture.nn_models import VelocityField
from src.utils.dataset import IsingSampler, GaussianBaseSampler

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Path to training data (adjust as needed)
    h5path = "../../data/gridstates_0.588_0.100.hdf5"

    # Create samplers
    sampler = IsingSampler(h5path, device=device)
    base = GaussianBaseSampler(channels=1, height=64, width=64)

    # Create flow and model
    flow = LinearFlow(sampler, base)
    model = VelocityField(L=64)

    z, _ = sampler.sample(10)        
    t = torch.rand(1,1,1,1).to(z)

    print(flow.sample_conditional_path_onehot(z, t).shape)

    # Create trainer
    trainer = IsingFlowTrainer(flow, model)

    # Train the model
    print("Starting training...")
    trainer.train_onehot(num_epochs=1000, device=device, lr=1e-3, batch_size=32)
    print("Training completed.")

    exit()

    # Optional: Sample some data after training
    print("Sampling from trained model...")
    with torch.no_grad():
        simulator = FlowSimulator(model)
        num_samples = 10
        num_steps = 100
        x_init = base.sample(num_samples, device=device)
        ts = torch.linspace(0, 1, num_steps + 1, device=device).repeat(num_samples, 1, 1, 1, 1)
        generated = simulator.simulate(x_init, ts)
        print(f"Generated samples shape: {generated.shape}")

if __name__ == "__main__":
    main()
