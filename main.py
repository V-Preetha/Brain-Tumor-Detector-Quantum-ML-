import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


# === ACTUAL PATH TO DATA ===
DATA_DIR = r"1\Brain Tumor Data Set\Brain Tumor Data Set" # can edit your path according to your dataset location

IMG_SIZE = (4,4) 


FOLDER_TUMOR = "Brain Tumor" # can change folder name according to your dataset
FOLDER_HEALTHY = "Healthy"  

# === Label Mapping ===
LABEL_MAP = {
    FOLDER_TUMOR: 1,
    FOLDER_HEALTHY: 0
}

# === Image Loader Function ===
def load_images_from_folder(folder_path, label):
    X, y = [], []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
            try:
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert("L")  # Grayscale
                img = img.resize(IMG_SIZE) #Resizing
                img = np.array(img).astype(np.float32) / 255.0 # converting to float32 just in case.
                X.append(img.flatten())
                y.append(label)
            except Exception as e:
                print(f"Skipping {filename}: {e}")
    return X, y

# === Load Data ===
X, y = [], []

for folder, label in LABEL_MAP.items():
    folder_path = os.path.join(DATA_DIR, folder)
    print(f"Looking in: {folder_path} (Label: {label})")

    if not os.path.exists(folder_path):
        print("Folder not found:", folder_path)
        continue

    X_part, y_part = load_images_from_folder(folder_path, label)
    X.extend(X_part)
    y.extend(y_part)

# === Final Arrays ===
X = np.array(X)
y = np.array(y)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Print Results ===
print("Data loaded successfully!")
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
print("Feature length per image:", X_train.shape[1])  # Should be 16

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Data saved as .npy files!")
