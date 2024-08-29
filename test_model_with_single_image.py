from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch
import torch.nn.functional as F

MODEL_PATH = r"C:\Workspace\Models\vit-base-happy-sad"  # Replace with your model path
IMAGE_PATH = r"C:\Workspace\Datasets\test_data\sad.png"  # Replace with your image path

# Load model and feature extractor  
model = ViTForImageClassification.from_pretrained(MODEL_PATH)
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_PATH)

# Load and preprocess image
image = Image.open(IMAGE_PATH).convert("RGB")  # Ensure image is in RGB format

# Extract features
inputs = feature_extractor(images=image, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move inputs to the same device as the model

# Perform inference
with torch.no_grad():
    logits = model(**inputs).logits

# Get predicted label
predicted_label_index = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_label_index]
print(f"Predicted label: {predicted_label}")

# Apply softmax to get probabilities
probabilities = F.softmax(logits, dim=-1)

# Multiply by 100 to get percentages
percentages = probabilities * 100

# Print the percentages for each class
for i, percentage in enumerate(percentages[0]):
    label = model.config.id2label[i]
    print(f"{label}: {percentage.item():.2f}%")