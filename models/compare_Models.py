import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import label_binarize

from dataloader.dataloader import get_dataloaders

from models.Models.model_LSTM import SignLanguageLSTM
from models.Models.model_GRU import SignLanguageGRU
from models.Models.model_BiLSTM import SignLanguageBiLSTM
from models.Models.model_CNN import SignLanguageCNN
from models.Models.model_CNNGRU import CNN_GRU
from models.Models.model_CNNBiLSTM import CNN_BiLSTM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")

ENCODER_PATH = os.path.join(BASE_DIR, "data", "objects", "label_encoder.pkl")

RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "results")

MODEL_DIR = os.path.join(BASE_DIR, "models", "Models")

os.makedirs(RESULTS_DIR, exist_ok=True)

_, _, test_loader = get_dataloaders(CSV_PATH, ENCODER_PATH, 32)

criterion = nn.CrossEntropyLoss()
# MODELS

models = {
    "LSTM": (SignLanguageLSTM(), "lstm_best.pth"),
    "GRU": (SignLanguageGRU(), "gru_best.pth"),
    "BiLSTM": (SignLanguageBiLSTM(), "bilstm_best.pth"),
    "CNN": (SignLanguageCNN(), "cnn_best.pth"),
    "CNN_GRU": (CNN_GRU(), "cnn_gru_best.pth"),
    "CNN_BiLSTM": (CNN_BiLSTM(), "cnn_bilstm_best.pth"),
}
# EVALUATE MODEL
def evaluate_model(model, path):

    path = os.path.join(MODEL_DIR, path)

    if not os.path.exists(path):
        print(f"❌ Missing: {path}")
        return None

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    start = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    end = time.time()
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    recall = recall_score(all_labels, all_preds, average='weighted') * 100
    precision = precision_score(all_labels, all_preds, average='weighted') * 100
    loss = total_loss / len(test_loader)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, loss, f1, recall, precision, (end - start), cm, np.array(all_probs), np.array(all_labels)

# MODEL COMPARISON
results = []
roc_data = {}
print("\n📊 Comparing Models...\n")
for name, (model, file) in models.items():
    res = evaluate_model(model, file)
    if res is None:
        continue
    acc, loss, f1, recall, precision, time_taken, cm, probs, labels = res

    print(f"{name} → Acc: {acc:.2f}% | F1: {f1:.2f}%")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Loss": loss,
        "Time (s)": time_taken
    })
    roc_data[name] = (labels, probs)
    np.save(os.path.join(RESULTS_DIR, f"{name}_confusion.npy"), cm)


# SAVE TABLE

df = pd.DataFrame(results)

print("\n📋 Comparison Table:\n")
print(df)

df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)


# LOAD HISTORY FILES

history_files = {
    "CNN": "cnn_history.pth",
    "CNN_GRU": "cnn_gru_history.pth",
    "CNN_BiLSTM": "cnn_bilstm_history.pth"
}

histories = {}

print("\n📈 Loading Training Histories...\n")

for model_name, history_file in history_files.items():
    history_path = os.path.join(MODEL_DIR, history_file)
    if os.path.exists(history_path):
        histories[model_name] = torch.load(history_path, map_location=DEVICE)
        print(f"✅ Loaded: {history_file}")

    else:
        print(f"❌ Missing: {history_file}")


# FIG 8.1
# OVERFITTING COMPARISON

fig, axes = plt.subplots(1, 3, figsize=(12, 5))
selected_models = ["CNN", "CNN_GRU", "CNN_BiLSTM"]
for idx, model_name in enumerate(selected_models):
    history = histories[model_name]
    epochs = range(1, len(history["train_acc"]) + 1)
    ax = axes[idx]
    # LOSS
    ax.plot(
        epochs,
        history["train_loss"],
        color='blue',
        linewidth=2,
        label="Train Loss"
    )
    ax.plot(
        epochs,
        history["val_loss"],
        color='orange',
        linestyle='--',
        linewidth=2,
        label="Val Loss"
    )
    # ACCURACY
    ax.plot(
        epochs,
        history["train_acc"],
        color='green',
        linewidth=2,
        label="Train Acc"
    )
    ax.plot(
        epochs,
        history["val_acc"],
        color='red',
        linestyle='--',
        linewidth=2,
        label="Val Acc"
    )
    ax.set_title(f"{model_name} - Train vs Validation")
    ax.set_xlabel("Epoch")
    ax.grid(True)
    ax.legend(fontsize=8)
plt.suptitle("OVERFITTING CHECK - Train vs Validation", fontsize=16)
plt.tight_layout()                    
plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "Overfitting_Comparison.png"
    )
)

plt.show()

# FIG 8.2
# ZOOMED ROC CURVE
plt.figure(figsize=(8, 6))
num_classes = len(np.unique(list(roc_data.values())[0][0]))
for name, (labels, probs) in roc_data.items():

    labels_bin = label_binarize(labels, classes=range(num_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        linewidth=2,
        label=f"{name} (AUC={roc_auc['micro']:.3f})"
    )
    
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 0.2])
plt.ylim([0.8, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Zoomed ROC Curve Comparison of Deep Learning Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "Zoomed_ROC.png"))
plt.show()

# FIG 8.3
# TRAINING VS VALIDATION LOSS
plt.figure(figsize=(12, 6))
for model_name, history in histories.items():

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.plot(
        epochs,
        history["train_loss"],
        linewidth=2,
        label=f"{model_name} Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        linestyle='--',
        linewidth=2,
        label=f"{model_name} Validation Loss"
    )

plt.title("Training and Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "Loss_Comparison.png"))
plt.show()
# FIG 8.4
# TRAINING VS VALIDATION ACCURACY
plt.figure(figsize=(12, 6))

for model_name, history in histories.items():

    epochs = range(1, len(history["train_acc"]) + 1)

    plt.plot(
        epochs,
        history["train_acc"],
        linewidth=2,
        label=f"{model_name} Train Accuracy"
    )

    plt.plot(
        epochs,
        history["val_acc"],
        linestyle='--',
        linewidth=2,
        label=f"{model_name} Validation Accuracy"
    )

plt.title("Training and Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "Accuracy_Comparison.png"))
plt.show()

# METRIC BAR GRAPH
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
df.set_index("Model")[metrics].plot(kind="bar", figsize=(10, 6))
plt.title("Model vs Metric Comparison")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=45)

for p in plt.gca().patches:

    plt.gca().annotate(
        f"{p.get_height():.1f}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha='center',
        va='bottom',
        fontsize=8
    )
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "model_vs_metric.png"))
plt.show()

# ACCURACY GRAPH
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy.png"))
plt.show()

# BEST MODEL
best = df.loc[df["Accuracy"].idxmax()]
print("\n Best Model:\n")
print(best)