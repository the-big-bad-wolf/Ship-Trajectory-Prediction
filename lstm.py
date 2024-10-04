import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), 50).to(x.device)
        c_0 = torch.zeros(1, x.size(0), 50).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    # Load preprocessed data
    features = pd.read_csv("features.csv").astype("float64")
    labels = pd.read_csv("labels.csv")

    # Print datatypes of features and labels
    print("Features data type:", features.dtypes)
    print("Labels data type:", labels.dtypes)

    print("Features:", features.head())

    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features.values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels.values, dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Define model, loss function, and optimizer
    input_size = features.shape[1]
    hidden_size = 50
    output_size = labels.shape[1]
    num_layers = 1

    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features.unsqueeze(1))
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save the model
    torch.save(model.state_dict(), "lstm_model.pth")
