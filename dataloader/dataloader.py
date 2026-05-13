# dataloader/dataloader.py

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import Counter
from dataloader.dataset import SignDataset


def get_dataloaders(csv_path, encoder_path, batch_size):

    dataset = SignDataset(csv_path, encoder_path)

    indices = list(range(len(dataset)))
    labels = dataset.labels

    # STEP 1: Train vs Temp 
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.3,
        stratify=labels,
        random_state=42
    )

    # STEP 2: Val vs Test (Stratified)
    temp_labels = labels[temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )

    # Create subsets
    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    test_ds  = Subset(dataset, test_idx)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"   Train : {len(train_ds)} samples")
    print(f"   Val   : {len(val_ds)} samples")
    print(f"   Test  : {len(test_ds)} samples")

    # DEBUG: check class distribution
    print("\n🔎 Class distribution (Train):", Counter(labels[train_idx]))
    print("🔎 Class distribution (Val):  ", Counter(labels[val_idx]))
    print("🔎 Class distribution (Test): ", Counter(labels[test_idx]))

    return train_loader, val_loader, test_loader