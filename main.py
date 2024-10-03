import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
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


# Custom Dataset class
class AISTrajectoryDataset(Dataset):
    def __init__(self, csv_file: str):
        self.data = pd.read_csv(csv_file, delimiter="|")
        self.data = self.__preprocess__(self.data)
        self.features = self.data.iloc[:, :-1].values
        self.labels = self.data.iloc[:, -1].values

    def __preprocess__(self, data: pd.DataFrame) -> pd.DataFrame:
        data["time"] = pd.to_datetime(data["time"])
        data["time"] = (data["time"] - data["time"].min()).dt.total_seconds() / 60

        navstat_dummies = pd.get_dummies(data["navstat"], prefix="navstat")
        data = pd.concat([data, navstat_dummies], axis=1)
        data.drop("navstat", axis=1, inplace=True)

        data.drop("etaRaw", axis=1, inplace=True)

        data.drop("portId", axis=1, inplace=True)

        print(data.head())
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.float32
        )


# Hyperparameters
input_size = 10  # Adjust based on your dataset
hidden_size = 50
num_layers = 2
output_size = 1
num_epochs = 20
batch_size = 32
learning_rate = 0.001

# Load dataset
dataset = AISTrajectoryDataset("task/ais_train.csv")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for features, labels in dataloader:
        outputs = model(features)
        loss = criterion(outputs, labels.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete.")
