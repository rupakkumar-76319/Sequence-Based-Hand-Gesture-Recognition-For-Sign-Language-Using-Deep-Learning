import os
from dataloader.dataloader import get_dataloaders

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_dir = os.path.join(BASE_DIR, "data", "processed")
encoder_path = os.path.join(BASE_DIR, "data", "objects", "label_encoder.pkl")

train_loader, val_loader, test_loader = get_dataloaders(
    data_dir=data_dir,
    encoder_path=encoder_path,
    batch_size=32
)

# Test one batch
for x, y in train_loader:
    print("Train batch:", x.shape, y.shape)
    break

for x, y in val_loader:
    print("Val batch:", x.shape, y.shape)
    break

for x, y in test_loader:
    print("Test batch:", x.shape, y.shape)
    break