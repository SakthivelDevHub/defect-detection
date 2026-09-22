"""Beginner-friendly CNN defect classifier.

This file has only two modes:

1. MODE = "train"
   Train the CNN from defect_detection_images.csv and save the best model.

2. MODE = "predict"
   Load the saved model, predict one PNG/JPG image, and display its label
   and confidence percentage.

Change the settings below, then run:
    python defect_cnn_pytorch.py
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# SETTINGS: change these values when needed
# ============================================================

MODE = "predict"  # Use "train" or "predict"

# Folder containing this Python script.
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "defect_detection_images.csv"
MODEL_PATH = BASE_DIR / "defect_cnn_best.pth"
IMAGE_PATH = BASE_DIR / "new_image.png"
IMAGE_SIZE = 32
NUMBER_OF_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
SEED = 42


# Use a GPU when one is available; otherwise use the CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed():
    """Make the split and training as repeatable as possible."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


# ============================================================
# CNN ARCHITECTURE
# ============================================================


class DefectCNN(nn.Module):
    def __init__(self, number_of_classes=NUMBER_OF_CLASSES):
        super().__init__()

        # These layers extract visual features from the image.
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # These layers use the extracted features to select a class.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, number_of_classes),
        )

    def forward(self, images):
        features = self.features(images)
        predictions = self.classifier(features)
        return predictions


# ============================================================
# DATA PREPARATION
# ============================================================


def load_and_prepare_data():
    """Load the CSV and return train, validation, and test arrays."""
    print("Loading dataset...")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH.resolve()}")

    dataframe = pd.read_csv(CSV_PATH)

    # Select only columns containing pixel values.
    pixel_columns = [
        column
        for column in dataframe.columns
        if column.startswith("pixel_")
    ]

    expected_pixel_count = IMAGE_SIZE * IMAGE_SIZE
    if len(pixel_columns) != expected_pixel_count:
        raise ValueError(
            f"Expected {expected_pixel_count} pixel columns, "
            f"but found {len(pixel_columns)}."
        )

    # Remove rows that cannot form a complete image.
    rows_before_cleaning = len(dataframe)
    dataframe = dataframe.dropna(
        subset=["label", "label_name"] + pixel_columns
    )
    dataframe = dataframe.drop_duplicates(subset="image_id")
    removed_rows = rows_before_cleaning - len(dataframe)

    print(f"Removed unusable rows: {removed_rows}")

    # Store class names in numeric-label order:
    # 0=normal, 1=scratch, 2=crack, 3=stain.
    label_table = (
        dataframe[["label", "label_name"]]
        .drop_duplicates()
        .sort_values("label")
    )
    class_names = label_table["label_name"].tolist()

    if len(class_names) != NUMBER_OF_CLASSES:
        raise ValueError(
            f"Expected {NUMBER_OF_CLASSES} classes, but found {class_names}."
        )

    # X contains image pixels. y contains correct numeric labels.
    X = dataframe[pixel_columns].to_numpy(dtype=np.float32)
    y = dataframe["label"].to_numpy(dtype=np.int64)

    # The supplied dataset is already scaled to 0-1. This condition also
    # supports a future dataset whose pixels are stored from 0-255.
    if X.max() > 1.0:
        X = X / 255.0

    # Convert each flat row of 1,024 pixels into a grayscale image tensor:
    # (number of images, 1 channel, 32 height, 32 width).
    X = X.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)

    # First split: 70% training and 30% temporary data.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )

    # Second split: divide temporary data into 15% validation and 15% test.
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp,
    )

    print(f"Training images:   {len(X_train)}")
    print(f"Validation images: {len(X_validation)}")
    print(f"Test images:       {len(X_test)}")
    print(f"Classes:           {class_names}")

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        class_names,
    )


def create_data_loader(X, y, shuffle):
    """Convert NumPy arrays into tensors and return batches of data."""
    image_tensor = torch.from_numpy(X)
    label_tensor = torch.from_numpy(y)

    dataset = TensorDataset(image_tensor, label_tensor)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


# ============================================================
# TRAINING AND VALIDATION
# ============================================================


def evaluate(model, data_loader, loss_function):
    """Calculate loss and accuracy without changing model weights."""
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = loss_function(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            correct_predictions += (predictions == labels).sum().item()
            total_images += labels.size(0)

    average_loss = total_loss / total_images
    accuracy = correct_predictions / total_images

    return average_loss, accuracy


def train_model():
    """Train the CNN, save the best model, and print final test accuracy."""
    set_seed()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        class_names,
    ) = load_and_prepare_data()

    train_loader = create_data_loader(X_train, y_train, shuffle=True)
    validation_loader = create_data_loader(
        X_validation,
        y_validation,
        shuffle=False,
    )
    test_loader = create_data_loader(X_test, y_test, shuffle=False)

    model = DefectCNN().to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float("inf")

    print(f"\nTraining on: {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()

        total_training_loss = 0.0
        correct_training_predictions = 0
        total_training_images = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Remove gradients calculated for the previous batch.
            optimizer.zero_grad()

            # Forward pass: make predictions.
            outputs = model(images)

            # Compare predictions with the correct labels.
            loss = loss_function(outputs, labels)

            # Backward pass: calculate gradients.
            loss.backward()

            # Update the model's weights.
            optimizer.step()

            predictions = outputs.argmax(dim=1)
            total_training_loss += loss.item() * labels.size(0)
            correct_training_predictions += (
                predictions == labels
            ).sum().item()
            total_training_images += labels.size(0)

        training_loss = total_training_loss / total_training_images
        training_accuracy = (
            correct_training_predictions / total_training_images
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            loss_function,
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train loss: {training_loss:.4f} | "
            f"train accuracy: {training_accuracy:.2%} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.2%}"
        )

        # Keep the model that performed best on validation data.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                },
                MODEL_PATH,
            )

    # Test the best saved model only after training is complete.
    best_model, _ = load_saved_model()
    _, test_accuracy = evaluate(
        best_model,
        test_loader,
        loss_function,
    )

    print(f"\nFinal test accuracy: {test_accuracy:.2%}")
    print(f"Saved model: {MODEL_PATH.resolve()}")


# ============================================================
# NEW-IMAGE PREDICTION
# ============================================================


def load_saved_model():
    """Load the trained CNN and its class names."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH.resolve()}\n"
            "Run the file with MODE = 'train' first."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]

    model = DefectCNN(
        number_of_classes=len(class_names)
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


def prepare_new_image(image_path):
    """Convert one PNG/JPG into the same format used during training."""
    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path.resolve()}"
        )

    image = Image.open(image_path)
    image = image.convert("L")
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)

    # (32, 32) → (1 image, 1 channel, 32 height, 32 width)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    return image_tensor.to(DEVICE)


def predict_image(image_path):
    """Return only the predicted label and confidence percentage."""
    model, class_names = load_saved_model()
    image_tensor = prepare_new_image(image_path)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = probabilities.max(dim=1)

    label_name = class_names[predicted_index.item()]
    confidence_percentage = confidence.item() * 100

    return {
        "label_name": label_name,
        "confidence_percentage": round(confidence_percentage, 2),
    }


# ============================================================
# START THE SELECTED MODE
# ============================================================


if __name__ == "__main__":
    if MODE == "train":
        train_model()
    elif MODE == "predict":
        result = predict_image(IMAGE_PATH)
        print(f"Prediction: {result['label_name']}")
        print(f"Confidence: {result['confidence_percentage']}%")
    else:
        raise ValueError("MODE must be either 'train' or 'predict'.")
