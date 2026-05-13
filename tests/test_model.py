import torch

from models.Models.model_LSTM import SignLanguageLSTM
from models.Models.model_GRU import SignLanguageGRU
from models.Models.model_BiLSTM import SignLanguageBiLSTM

SEQ_LENGTH = 30
INPUT_SIZE = 63
NUM_CLASSES = 36

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = input("Choose model (lstm / gru / bilstm): ").lower()

if model_name == "lstm":
    model = SignLanguageLSTM(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)

elif model_name == "gru":
    model = SignLanguageGRU(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)

elif model_name == "bilstm":
    model = SignLanguageBiLSTM(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)

else:
    raise ValueError("❌ Invalid model name")

model = model.to(DEVICE)

x = torch.randn(32, SEQ_LENGTH, INPUT_SIZE).to(DEVICE)

with torch.no_grad():
    output = model(x)

print(f"Model: {model_name.upper()}")
print("Output shape:", output.shape)