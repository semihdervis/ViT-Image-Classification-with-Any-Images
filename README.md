# ViT-Image-Classification-with-Any-Images

## Overview
This repository provides a pipeline for fine-tuning a Vision Transformer (ViT) model on custom image datasets using Hugging Face's Transformers library. The code is designed to offer flexibility in dataset management, model fine-tuning, and inference, making it easy to adapt the ViT model to various image classification tasks

## Setup

### Clone the Repository
```bash
git clone https://github.com/semihdervis/ViT-Image-Classification-with-Any-Images.git
cd ViT-Image-Classification-with-Any-Images
```

### Install Requirements
Ensure you have Python 3.8+ installed. Install the necessary packages using `pip`:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. **Set Dataset and Output Directory:**
   - Replace `DATASET_PATH` in [`train.py`](train.py) with the path to your image dataset.
   - Set `OUTPUT_DIR` to your desired model output directory.

2. **Run Training:**
   ```bash
   python train.py
   ```

### Testing the Model with a Single Image

1. **Set Model and Image Paths:**
   - In [`test_model_with_single_image.py`](test_model_with_single_image.py), replace `MODEL_PATH` with the path to your trained model.
   - Replace `IMAGE_PATH` with the path to the image you want to classify.

2. **Run the Inference Script:**
   ```bash
   python test_model_with_single_image.py
   ```

## License
This project is licensed under the [MIT License](LICENSE).
