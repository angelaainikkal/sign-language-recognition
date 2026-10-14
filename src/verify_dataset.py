
import numpy as np
import json
from pathlib import Path

# Find the dataset files
data_folder = Path(__file__).parent.parent / "data"

images = np.load(data_folder / "images.npy")
labels = np.load(data_folder / "labels.npy")

with open(data_folder / "dataset_report.json", "r") as file:
    report = json.load(file)

print("DATASET VERIFICATION")
print("--------------------")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Number of images:", len(images))
print("Number of labels:", len(labels))
print("Image data type:", images.dtype)
print("Label data type:", labels.dtype)
print("Classes in report:", report["classes"])
print("Processed images in report:", report["processed_images"])
print("Failed files:", report["failed_files"])

print("\nFirst 10 labels:", labels[:10])
print("Unique labels:", np.unique(labels))
