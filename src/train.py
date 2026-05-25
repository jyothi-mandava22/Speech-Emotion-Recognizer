import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from dataset import build_dataframe
from extract_features import batch_extract
from model import build_model

EMOTIONS = ["neutral", "calm", "happy", "sad",
            "angry", "fearful", "disgust", "surprised"]

def plot_training_history(history, output_dir="models/"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved training_history.png")

def plot_confusion_matrix(y_test, y_pred, output_dir="models/"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{output_dir}confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved confusion_matrix.png")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    print("Loading dataset...")
    df = build_dataframe("data/raw")

    print("Extracting features...")
    X, y = batch_extract(df, cache_path="data/features.npy")
    print(f"Feature shape: {X.shape}")

    # Add channel dimension
    X = X[..., np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")

    print("Building model...")
    model = build_model(input_shape=X_train.shape[1:3])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=15, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=7, factor=0.5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            "models/ser_model.keras", save_best_only=True, verbose=1)
    ]

    print("Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=80,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    print("\nEvaluating...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred,
                                target_names=EMOTIONS,
                                zero_division=0))

    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)

    print("\nDone. Model saved to models/ser_model.keras")