import torch
import numpy as np
from PIL import Image
from qnn_model import HybridQNN  # Make sure this file is in the same folder

# === Image Settings ===
IMG_SIZE = (5,5)  # Same as training size

# === Load the trained model ===
model = HybridQNN()
model.load_state_dict(torch.load("trained_qnn_model.pth", map_location=torch.device("cpu")))
model.eval()

# === Image Preprocessing Function ===
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("L")  # Grayscale
        img = img.resize(IMG_SIZE)
        img = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img.flatten()).unsqueeze(0)  # Shape: [1, 16]
        return img_tensor
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# === Prediction Function ===
def predict(image_path):
    input_tensor = preprocess_image(image_path)
    if input_tensor is None:
        return

    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output >= 0.5).item()
        confidence = output.item()

    label = "Tumor" if prediction == 1 else "Healthy"
    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")

# === MAIN ===
if __name__ == "__main__":
    # === Your test image path here ===
    image_path = input("Enter path to brain scan image: ").strip()
    predict(image_path)
