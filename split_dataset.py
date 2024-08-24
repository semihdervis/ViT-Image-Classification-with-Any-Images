from transformers import ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, load_metric
import numpy as np
import torch
from PIL import Image

def split_dataset(ds):
    
    # Check for existing splits and handle accordingly
    if 'train' in ds.keys() and 'validation' in ds.keys() and 'test' in ds.keys():
        # Dataset already has train, validation, and test splits
        dataset = DatasetDict({
            'train': ds['train'],
            'validation': ds['validation'],
            'test': ds['test']
        })
    elif 'train' in ds.keys() and 'validation' in ds.keys():
        # Dataset has only train and validation splits, so create a test split from validation
        ds_split = ds['validation'].train_test_split(test_size=0.5)
        dataset = DatasetDict({
            'train': ds['train'],
            'validation': ds_split['train'],
            'test': ds_split['test']
        })
    elif 'train' in ds.keys() and 'test' in ds.keys():
        # Dataset has only train and test splits, so create a validation split from train
        train_val = ds['train'].train_test_split(test_size=0.1)
        dataset = DatasetDict({
            'train': train_val['train'],
            'validation': train_val['test'],
            'test': ds['test']
        })
    elif 'train' in ds.keys():
        # Dataset only has a train split, so create both validation and test splits
        ds_split = ds['train'].train_test_split(test_size=0.1)  # Split 10% for test set
        train_val = ds_split['train'].train_test_split(test_size=0.2857)  # Split the remaining 90% into train and validation (0.2 / 0.9 ≈ 0.2857)
        dataset = DatasetDict({
            'train': train_val['train'],
            'validation': train_val['test'],
            'test': ds_split['test']
        })
    else:
        raise ValueError("The dataset does not have a 'train' split.")

    return dataset

