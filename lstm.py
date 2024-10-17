import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn


# Define the dataset
class ShipDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.float32
        )


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


# Define the custom loss function
class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute geodesic distance using vectorized operations
        lat1, lon1 = outputs[:, 0], outputs[:, 1]
        lat2, lon2 = targets[:, 0], targets[:, 1]

        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(torch.deg2rad, (lat1, lon1, lat2, lon2))

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            torch.sin(dlat / 2) ** 2
            + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        )
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        R = 6371  # Radius of Earth in kilometers
        geodesic_distance = R * c

        # Square geodesic distance
        geodesic_distance_squared = geodesic_distance**2

        # Mean squared error for the rest of the variables
        mse = nn.functional.mse_loss(outputs[:, 2:], targets[:, 2:])

        # Combine geodesic distance squared with MSE
        loss = geodesic_distance_squared.mean() * 1000 + mse
        return loss


if __name__ == "__main__":

    # Load the data
    features = pd.read_csv("data/features.csv").values
    labels = pd.read_csv("data/labels.csv").values

    # Hyperparameters
    input_size = features.shape[1]
    hidden_size = 50
    num_layers = 2
    output_size = labels.shape[1]
    learning_rate = 0.001
    num_epochs = 1000
    batch_size = 32

    # Load data
    dataset = ShipDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = GeodesicLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for features, labels in dataloader:
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        if (epoch + 1) % 50 == 0:
            torch.save(
                model.state_dict(), f"models/output/lstm_model_epoch_{epoch+1}.pth"
            )
