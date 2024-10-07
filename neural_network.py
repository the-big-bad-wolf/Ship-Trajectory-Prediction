import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


class ShipTrajectoryMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ShipTrajectoryMLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.uppers = torch.tensor([90, 180], dtype=torch.float32)
        self.lowers = torch.tensor([-90, -180], dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        # Clip the latitude and longitude
        out[:, 0] = torch.clamp(out[:, 0].clone(), self.lowers[0], self.uppers[0])
        out[:, 1] = torch.clamp(out[:, 1].clone(), self.lowers[1], self.uppers[1])
        return out


class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
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
        R = 6371000  # Radius of Earth in meters
        geodesic_distance = R * c

        square_geodesic_distance = geodesic_distance.square()

        mse_loss = nn.functional.mse_loss(
            outputs[:, 2:], targets[:, 2:], reduction="none"
        ).mean(dim=1)

        total_loss = square_geodesic_distance + mse_loss
        return total_loss.mean()


if __name__ == "__main__":
    # Load preprocessed data
    features = pd.read_csv("features.csv").astype("float32")
    labels = pd.read_csv("labels.csv").astype("float32")

    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features.values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels.values, dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    # Define model, loss function, and optimizer
    input_size = features.shape[1]
    hidden_size = 16
    output_size = labels.shape[1]

    model = ShipTrajectoryMLP(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load("mlp_model_200.pth"))
    loss_fn = GeodesicLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 200
    model.train()
    for epoch in range(num_epochs):
        for features_batch, labels_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(features_batch)
            loss = loss_fn(outputs, labels_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        if (epoch + 1) % 25 == 0:
            torch.save(model.state_dict(), f"models/mlp_model_epoch_{epoch+1}.pth")

    # Save the model
    torch.save(model.state_dict(), "mlp_model.pth")
