
import torch
import torch.nn as nn

class TransactionAutoencoder(nn.Module):
    """Small autoencoder for tabular transaction features. Bottleneck forces
    the model to learn a compressed representation of NORMAL transactions —
    fraud/anomalies reconstruct poorly, and that reconstruction error becomes
    the anomaly score."""
    def __init__(self, input_dim, bottleneck_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12), nn.ReLU(),
            nn.Linear(12, 8), nn.ReLU(),
            nn.Linear(8, bottleneck_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 8), nn.ReLU(),
            nn.Linear(8, 12), nn.ReLU(),
            nn.Linear(12, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def reconstruction_error(model, X_tensor):
    """Per-row MSE between input and reconstruction — the anomaly score."""
    with torch.no_grad():
        recon = model(X_tensor)
        errors = torch.mean((X_tensor - recon) ** 2, dim=1)
    return errors.cpu().numpy()
