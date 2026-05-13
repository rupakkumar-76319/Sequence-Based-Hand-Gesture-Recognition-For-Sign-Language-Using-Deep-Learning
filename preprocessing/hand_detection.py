import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os

# =========================
# [1] Choose Mode
# =========================
mode = input("Choose mode (alphabet / number / words): ").strip().lower()

if mode == "alphabet":
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

elif mode == "number":
    labels = list("0123456789")

elif mode == "words":
    labels = ["hello", "hi", "eat", "drink"]

else:
    print("Invalid mode! Defaulting to alphabet.")
    mode = "alphabet"
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# =========================
# [2] Create Folder Structure
# =========================
for label in labels:
    os.makedirs(f"data/raw/{mode}/{label}", exist_ok=True)

# =========================
# [3] Video Setup
# =========================
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# =========================
# [4] Sequence Setup
# =========================
sequence = []
sequence_length = 30
collecting = False
current_label = None

# =========================
# [5] Main Loop
# =========================
while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    key = cv2.waitKey(1) & 0xFF

    # =========================
    # [6] Label Selection
    # =========================
    if mode == "alphabet" and 65 <= key <= 90:
        current_label = chr(key)
        print(f"Selected Label: {current_label}")

    elif mode == "number" and 48 <= key <= 57:
        current_label = chr(key)
        print(f"Selected Label: {current_label}")

    elif mode == "words":
        # simple mapping (1–4 keys)
        word_map = {
            ord('1'): "hello",
            ord('2'): "hi",
            ord('3'): "eat",
            ord('4'): "drink"
        }
        if key in word_map:
            current_label = word_map[key]
            print(f"Selected Label: {current_label}")

    # =========================
    # [7] Start Collecting
    # =========================
    elif key == ord('s') and current_label is not None:
        collecting = True
        sequence = []
        print(f"Collecting for: {current_label}")

    # Exit
    elif key == 27:
        break

    # =========================
    # [8] Hand Processing
    # =========================
    if hands:
        hand = hands[0]

        lmList = np.array(hand['lmList'])

        # 🔥 Normalize (important)
        lmList = lmList - lmList[0]

        # 🔥 Feature vector (correct)
        feature_vector = lmList.flatten()

        # =========================
        # [9] Sequence Build
        # =========================
        if collecting:
            sequence.append(feature_vector)

            print(f"Collecting: {len(sequence)}/{sequence_length}")

            if len(sequence) == sequence_length:
                save_dir = f"data/raw/{mode}/{current_label}"
                count = len(os.listdir(save_dir))

                save_path = os.path.join(save_dir, f"{count}.npy")
                np.save(save_path, np.array(sequence))

                print(f"Saved: {save_path}")

                sequence = []
                collecting = False

    # =========================
    # [10] UI Display
    # =========================
    cv2.putText(img, f"Mode: {mode}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(img, f"Label: {current_label}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, f"Collecting: {collecting}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # =========================
    # [11] Show Frame
    # =========================
    cv2.imshow("Image", img)

# =========================
# [12] Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()