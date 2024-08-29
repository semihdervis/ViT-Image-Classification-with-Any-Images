import cv2
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

MODEL_PATH = r"C:\Workspace\Models\vit-base-happy-sad"  # Replace with your model path

# Load model and feature extractor  
model = ViTForImageClassification.from_pretrained(MODEL_PATH)
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_PATH)

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Extract features
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move inputs to the same device as the model

    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get predicted label
    predicted_label_index = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_label_index]

    # Apply softmax to get probabilities
    probabilities = F.softmax(logits, dim=-1)

    # Multiply by 100 to get percentages
    percentages = probabilities * 100

    # Display the predicted label on the frame
    cv2.putText(frame, f"Predicted label: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the percentages for each class
    y_offset = 60
    for i, percentage in enumerate(percentages[0]):
        label = model.config.id2label[i]
        text = f"{label}: {percentage.item():.2f}%"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y_offset += 20

    # Show the frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()