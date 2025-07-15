import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from qnn_model import HybridQNN  # Your Step 2 model file
from torch.optim.lr_scheduler import StepLR

# === Load preprocessed data ===
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")


print("X_train shape:", X_train.shape)
print("X_train ndim:", X_train.ndim)

# === Convert to PyTorch tensors ===
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# === Create DataLoader ===
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# === Initialize model, loss, optimizer ===
model = HybridQNN()
criterion = nn.BCELoss()


optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # reduces LR every 10 epochs

# === Training loop ===
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.float()
        batch_y = batch_y.float().view(-1,1)


        optimizer.zero_grad()
        outputs = model(batch_x)

        loss = criterion(outputs, batch_y.view(-1, 1))

        loss.backward()
        optimizer.step()
        


        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    accuracy = correct / total * 100
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.2f}%")

# === Save trained model ===
torch.save(model.state_dict(), "trained_qnn_model.pth")
print("Model saved as trained_qnn_model.pth")

