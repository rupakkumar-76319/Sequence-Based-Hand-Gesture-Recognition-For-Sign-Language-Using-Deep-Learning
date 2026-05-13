import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

from dataloader.dataloader import get_dataloaders

from models.Models.model_LSTM import SignLanguageLSTM
from models.Models.model_GRU import SignLanguageGRU
from models.Models.model_BiLSTM import SignLanguageBiLSTM
from models.Models.model_CNN import SignLanguageCNN
from models.Models.model_CNNGRU import CNN_GRU
from models.Models.model_CNNBiLSTM import CNN_BiLSTM

# SEED
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# MODEL SELECT
model_name = input("Choose model (lstm / gru / bilstm / cnn / cnn_gru / cnn_bilstm): ").lower()

models_dict = {
    "lstm": SignLanguageLSTM(),
    "gru": SignLanguageGRU(),
    "bilstm": SignLanguageBiLSTM(),
    "cnn": SignLanguageCNN(),
    "cnn_gru": CNN_GRU(),
    "cnn_bilstm": CNN_BiLSTM()
}

if model_name not in models_dict:
    raise ValueError("Invalid model")

model = models_dict[model_name]

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
ENCODER_PATH = os.path.join(BASE_DIR, "data", "objects", "label_encoder.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_SAVE = os.path.join(BASE_DIR, f"models/Models/{model_name}_best.pth")

EPOCHS = 50
BATCH_SIZE = 32
LR = 0.0005

train_loader, val_loader, test_loader = get_dataloaders(
    CSV_PATH, ENCODER_PATH, BATCH_SIZE
)

model = model.to(DEVICE)

# ✅ Compute class weights ONLY from training data
train_labels = [y.item() for _, y in train_loader.dataset]
class_counts = np.bincount(train_labels)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

best_val = 0

train_losses, val_losses = [], []
train_accs, val_accs = [], []

# TRAIN LOOP
for epoch in range(EPOCHS):
    model.train()
    correct, total, running_loss = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, pred = torch.max(out, 1)
        total += y.size(0)
        correct += (pred == y).sum().item()

    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)

    # VALIDATION
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
            loss = criterion(out, y)

            val_loss += loss.item()

            _, pred = torch.max(out, 1)
            val_total += y.size(0)
            val_correct += (pred == y).sum().item()

    val_acc = 100 * val_correct / val_total
    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"{model_name.upper()} | Epoch {epoch+1} | Train {train_acc:.2f}% | Val {val_acc:.2f}%")

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), MODEL_SAVE)
        print("✅ Best model saved")

    scheduler.step()

# TEST
model.load_state_dict(torch.load(MODEL_SAVE))
model.eval()

test_correct, test_total = 0, 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        _, pred = torch.max(out, 1)

        test_total += y.size(0)
        test_correct += (pred == y).sum().item()

print(f"\n🎯 Final Test Accuracy: {100 * test_correct / test_total:.2f}%")

# ✅ SAVE CURVES
plt.figure()
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.xlabel(r"Epochs $\rightarrow$")
plt.ylabel(r"Accuracy $\rightarrow$")
plt.legend()
plt.title("Epochs vs Accuracy Curve")
plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_accuracy_curve.png"))
plt.close()

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel(r"Epochs $\rightarrow$")
plt.ylabel(r"Loss $\rightarrow$")
plt.legend()
plt.title("Loss Curve")
plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_loss_curve.png"))
plt.close()

history = {
    "train_acc": train_accs,
    "val_acc": val_accs,
    "train_loss": train_losses,
    "val_loss": val_losses
}

torch.save(
    history,
    os.path.join(BASE_DIR, f"models/Models/{model_name}_history.pth")
)

print("✅ Training history saved")

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel(r"Epochs $\rightarrow$")
plt.ylabel(r"Loss $\rightarrow$")
plt.legend()
plt.title("Loss Curve")
plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_loss_curve.png"))
plt.close()

# SAVE TRAINING HISTORY

history = {
    "train_acc": train_accs,
    "val_acc": val_accs,
    "train_loss": train_losses,
    "val_loss": val_losses
}

torch.save(
    history,
    os.path.join(BASE_DIR, f"models/Models/{model_name}_history.pth")
)

print("✅ Training history saved")