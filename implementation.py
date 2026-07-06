import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define a simple dataset for transportation demand prediction
class TransportationDataset(Dataset):
    def __init__(self, num_samples=1000):
        np.random.seed(42)
        self.features = np.random.rand(num_samples, 3)  # Example features: [time_of_day, day_of_week, weather_condition]
        self.labels = (self.features[:, 0] * 0.5 + self.features[:, 1] * 0.3 + self.features[:, 2] * 0.2 + np.random.rand(num_samples) * 0.1).reshape(-1, 1)  # Example demand

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Define a simple neural network for demand prediction
class DemandPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DemandPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Training function
def train_model(model, dataloader, criterion, optimizer, epochs=20):
    for epoch in range(epochs):
        total_loss = 0.0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

# Testing function
def test_model(model, dataloader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            predictions.append(outputs.numpy())
            actuals.append(labels.numpy())
    return np.vstack(predictions), np.vstack(actuals)

if __name__ == '__main__':
    # Hyperparameters
    input_size = 3
    hidden_size = 16
    output_size = 1
    batch_size = 32
    learning_rate = 0.01
    epochs = 20

    # Create dataset and dataloaders
    dataset = TransportationDataset(num_samples=1000)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = DemandPredictionModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs)

    # Test the model
    predictions, actuals = test_model(model, test_loader)
    print("Sample Predictions vs Actuals:")
    for pred, actual in zip(predictions[:5], actuals[:5]):
        print(f"Predicted: {pred[0]:.4f}, Actual: {actual[0]:.4f}")