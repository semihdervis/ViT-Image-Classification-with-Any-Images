from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch
import torch.nn.functional as F

# Load model and feature extractor
model_path = "vit-base-cat-dog"  # Replace with the path to your model
model = ViTForImageClassification.from_pretrained(model_path)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)

# Load and preprocess image
image_path = r"test_data/dog.png"  # Replace with your image path
image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format

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