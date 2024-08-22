from transformers import ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, load_metric
import numpy as np
import torch
from PIL import Image

DATASET_PATH = "diabetic-retinopathy"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess dataset
ds = load_dataset(DATASET_PATH).rename_column('label', 'labels')
ds = ds['train'].train_test_split(test_size=0.1)
train_val = ds['train'].train_test_split(test_size=0.1)
ds['train'], ds['validation'], ds['test'] = train_val['train'], train_val['test'], ds['test']
dataset = DatasetDict(ds)

# Initialize feature extractor and model
model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
labels = ds['train'].features['labels'].names
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
).to(device)

# Transform function
def transform(example_batch):
    images = [np.moveaxis(np.array(x.convert('RGB')), -1, 0) for x in example_batch['image']]
    inputs = feature_extractor(images, return_tensors='pt')
    inputs['labels'] = example_batch['labels']
    return inputs

# Prepare dataset
prepared_ds = ds.with_transform(transform)

# Data collator
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

# Metric and compute function
metric = load_metric("accuracy", trust_remote_code=True)
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

# Training arguments
training_args = TrainingArguments(
    output_dir="./vit-base--v5",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=feature_extractor,
)

# Train and evaluate
train_results = trainer.train()
trainer.save_model()
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_ds['test'])
trainer.save_metrics("eval", metrics)

# Model card creation
kwargs = {
    "finetuned_from": model.config._name_or_path,
    "tasks": "image-classification",
    "dataset": 'custom brats layers',
    "tags": ['image-classification'],
}
if training_args.push_to_hub:
    trainer.push_to_hub('🍻 cheers', **kwargs)
else:
    trainer.create_model_card(**kwargs)