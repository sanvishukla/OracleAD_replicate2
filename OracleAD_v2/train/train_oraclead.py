import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.windowing import make_windows


# --------------------------------------------------
# OracleAD Model (minimal, stable)
# --------------------------------------------------
class OracleAD(nn.Module):
    def __init__(self, num_vars, window_size, hidden_dim=64):
        super().__init__()

        self.encoder = nn.GRU(
            input_size=num_vars,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.decoder = nn.Linear(hidden_dim, num_vars)

    def forward(self, x):
        _, h = self.encoder(x)
        h = h.squeeze(0)
        y_hat = self.decoder(h)
        return y_hat, h


# --------------------------------------------------
# TRAIN FUNCTION
# --------------------------------------------------
def train_oraclead(
    train_csv,
    window_size=20,
    batch_size=64,
    epochs=10,
    lr=1e-3,
    device="cpu"
):
    """
    Trains OracleAD and returns trained model + SLS matrix.
    """

    # -------------------------
    # Load & normalize data
    # -------------------------
    df = pd.read_csv(train_csv)
    data = df.drop(columns=["timestamp_(min)"]).values

    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-6
    data = (data - mean) / std

    data = np.nan_to_num(data)

    # -------------------------
    # Windowing
    # -------------------------
    X = make_windows(data, window_size=window_size)
    X = torch.tensor(X, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # -------------------------
    # Model
    # -------------------------
    model = OracleAD(
        num_vars=X.shape[-1],
        window_size=window_size
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # -------------------------
    # Training loop
    # -------------------------
    print("\n=== Training OracleAD ===")

    model.train()
    hidden_states = []

    for epoch in range(epochs):
        epoch_loss = []

        for (x,) in loader:
            x = x.to(device)

            target = x[:, -1, :]
            y_hat, h = model(x)

            loss = criterion(y_hat, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            hidden_states.append(h.detach().cpu().numpy())

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Avg MSE Loss: {np.mean(epoch_loss):.6f}"
        )

    # -------------------------
    # Structural Latent Space
    # -------------------------
    H = np.concatenate(hidden_states, axis=0)
    SLS = np.cov(H.T)

    print("Training complete.")
    print("SLS shape:", SLS.shape)

    return model, SLS, mean, std
