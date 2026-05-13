import os
import numpy as np

INPUT_DIR  = "data/raw"
OUTPUT_DIR = "data/processed"

SEQUENCE_LENGTH = 30
NUM_LANDMARKS   = 21
DIM             = 3


# NORMALIZATION (IMPROVED)
def normalize_sequence(seq):
    # seq: (30, 21, 3)

    # 1. Translate (wrist)
    seq = seq - seq[:, 0:1, :]

    # 2. Scale (robust)
    scale = np.linalg.norm(seq, axis=(1, 2), keepdims=True)
    seq = seq / (scale + 1e-6)

    return seq


# VELOCITY
def add_velocity(seq):
    velocity = np.diff(seq, axis=0, prepend=seq[0:1])
    return np.concatenate([seq, velocity], axis=2)  


# MIRROR
def mirror_sequence(seq):
    seq = seq.copy()
    seq[:, :, 0] = -seq[:, :, 0]
    return seq


# FLATTEN
def flatten_sequence(seq):
    return seq.reshape(SEQUENCE_LENGTH, -1)


def process_dataset():
    total = 0

    for label in os.listdir(INPUT_DIR):
        label_path = os.path.join(INPUT_DIR, label)

        if not os.path.isdir(label_path):
            continue

        output_label_path = os.path.join(OUTPUT_DIR, label)
        os.makedirs(output_label_path, exist_ok=True)

        for file in os.listdir(label_path):
            if not file.endswith(".npy"):
                continue

            file_path = os.path.join(label_path, file)
            data = np.load(file_path)

            if data.shape != (SEQUENCE_LENGTH, NUM_LANDMARKS * DIM):
                print(f"Skipping {file} — wrong shape: {data.shape}")
                continue

            seq = data.reshape(SEQUENCE_LENGTH, NUM_LANDMARKS, DIM)

            # ORIGINAL
            norm_seq = normalize_sequence(seq)
            vel_seq  = add_velocity(norm_seq)
            final_seq = flatten_sequence(vel_seq)

            base_name = file.replace(".npy", "")
            np.save(os.path.join(output_label_path, base_name + ".npy"), final_seq)

            # MIRROR (after normalization)
            mirror_seq = mirror_sequence(norm_seq)
            mirror_vel = add_velocity(mirror_seq)
            mirror_final = flatten_sequence(mirror_vel)

            np.save(os.path.join(output_label_path, base_name + "_mirror.npy"), mirror_final)

            total += 1
            if total % 100 == 0:
                print(f"Processed {total} files...")

    print(f"\nDone! Total processed: {total}")


if __name__ == "__main__":
    process_dataset()