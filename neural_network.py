import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from geopy.distance import geodesic


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class ShipTrajectoryMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ShipTrajectoryMLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs[:, :2]
        targets = targets[:, :2]
        loss = torch.tensor(
            [
                geodesic((lat1, lon1), (lat2, lon2)).m
                for (lat1, lon1), (lat2, lon2) in zip(inputs, targets)
            ]
        )
        return loss.mean().requires_grad_()


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
    num_layers = 3

    model = ShipTrajectoryMLP(input_size, hidden_size, output_size)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 200
    model.train()
    for epoch in range(num_epochs):
        for features_batch, labels_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(features_batch.unsqueeze(1))
            loss = loss_fn(outputs, labels_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # Save the model
    torch.save(model.state_dict(), "lstm_model.pth")
