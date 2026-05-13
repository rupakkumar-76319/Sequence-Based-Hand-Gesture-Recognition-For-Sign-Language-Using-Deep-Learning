import os
import pickle
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
SAVE_DIR = os.path.join(BASE_DIR, "data", "objects")
SAVE_PATH = os.path.join(SAVE_DIR, "label_encoder.pkl")

os.makedirs(SAVE_DIR, exist_ok=True)

labels = []

# Read class folders (sorted = stable mapping)
for label in sorted(os.listdir(DATA_DIR)):
    label_path = os.path.join(DATA_DIR, label)

    if os.path.isdir(label_path) and label.strip() != "":
        labels.append(label)

# Create encoder
encoder = LabelEncoder()
encoder.fit(labels)

# Save encoder
with open(SAVE_PATH, "wb") as f:
    pickle.dump(encoder, f)

print("✅ Label encoder created successfully!")
print("Classes:", encoder.classes_)