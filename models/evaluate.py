import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

from dataloader.dataloader import get_dataloaders
from models.Models.model_LSTM import SignLanguageLSTM
from models.Models.model_GRU import SignLanguageGRU
from models.Models.model_BiLSTM import SignLanguageBiLSTM
from models.Models.model_CNN import SignLanguageCNN
from models.Models.model_CNNGRU import CNN_GRU
from models.Models.model_CNNBiLSTM import CNN_BiLSTM


# MODEL SELECTION
model_name = input("Choose model (lstm / gru / bilstm / cnn / cnn_gru / cnn_bilstm): ").lower()

if model_name == "lstm":
    model = SignLanguageLSTM()
elif model_name == "gru":
    model = SignLanguageGRU()
elif model_name == "bilstm":
    model = SignLanguageBiLSTM()
elif model_name == "cnn":
    model = SignLanguageCNN()
elif model_name == "cnn_gru":
    model = CNN_GRU()
elif model_name == "cnn_bilstm":
    model = CNN_BiLSTM()
else:
    raise ValueError("Invalid model")


# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
ENCODER_PATH = os.path.join(BASE_DIR, "data", "objects", "label_encoder.pkl")
MODEL_PATH = os.path.join(BASE_DIR, f"models/Models/{model_name}_best.pth")

RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# LOAD DATA
print("Loading test data...")
_, _, test_loader = get_dataloaders(CSV_PATH, ENCODER_PATH, batch_size=32)


# LOAD MODEL
print("Loading model...")
model = model.to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

num_classes = 36
class_names = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]

all_preds = []
all_labels = []
all_probs = []


# EVALUATION
print("Running evaluation...")

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)

        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())


all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)


# ACCURACY
accuracy = np.mean(all_preds == all_labels)
print(f"\n🎯 Test Accuracy: {accuracy * 100:.2f}%")


# CLASSIFICATION REPORT
print("\n📊 Classification Report:")
report = classification_report(all_labels, all_preds)
print(report)

with open(os.path.join(RESULTS_DIR, f"{model_name}_classification_report.txt"), "w") as f:
    f.write(report)


# CONFUSION MATRIX
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues")
plt.title(f"{model_name.upper()} Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

cm_path = os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

print(f"\n📁 Confusion matrix saved at: {cm_path}")


# ROC CURVE
y_true_bin = label_binarize(all_labels, classes=list(range(num_classes)))

plt.figure(figsize=(8, 6))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"{model_name.upper()} ROC Curve")
plt.legend(fontsize=6)

roc_path = os.path.join(RESULTS_DIR, f"{model_name}_roc_curve.png")
plt.savefig(roc_path)
plt.close()

print(f"📈 ROC curve saved at: {roc_path}")


# ✅ NEW: PER-CLASS FPR
fpr_list = []

for i in range(num_classes):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    if (FP + TN) == 0:
        fpr = 0.0
    else:
        fpr = FP / (FP + TN)

    fpr_list.append(fpr)


# FPR TABLE
df = pd.DataFrame({
    "Class": class_names,
    "FPR": fpr_list
})

print("\n📊 Per-Class FPR Table:")
print(df.to_string(index=False))

csv_path = os.path.join(RESULTS_DIR, f"{model_name}_fpr_table.csv")
df.to_csv(csv_path, index=False)

print(f"📁 FPR table saved at: {csv_path}")

top_k = 3
topk_preds = np.argsort(all_probs, axis=1)[:, -top_k:]
topk_acc = np.mean([all_labels[i] in topk_preds[i] for i in range(len(all_labels))])
print(f"\n🎯 Top-{top_k} Accuracy: {topk_acc*100:.2f}%")

# LATENCY
sample_input = x[:1].to(DEVICE)

start = time.time()
_ = model(sample_input)
end = time.time()

latency = (end - start) * 1000
print(f"⚡ Inference Latency: {latency:.2f} ms")

#Precision-Recall Curve
plt.figure()
for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
    plt.plot(recall, precision, label=f"{class_names[i]}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(fontsize=6)

plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_pr_curve.png"))
plt.close()

# FPR GRAPH
plt.figure(figsize=(14, 6))
plt.bar(range(num_classes), fpr_list)

plt.xticks(range(num_classes), class_names, rotation=90)
plt.xlabel("Classes")
plt.ylabel("False Positive Rate")
plt.title(f"{model_name.upper()} Per-Class FPR")

plt.tight_layout()

fpr_plot_path = os.path.join(RESULTS_DIR, f"{model_name}_fpr_graph.png")
plt.savefig(fpr_plot_path)
plt.close()

print(f"📈 FPR graph saved at: {fpr_plot_path}")


print("\n✅ Evaluation complete!")