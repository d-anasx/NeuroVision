import os
import time
import json
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
import tensorflow as tf
from mlflow import MlflowClient


def load_model():
    #loading the model
    CHECKPOINT_DIR = "/home/user/Sync/backup/checkpoints"
    model = tf.keras.models.load_model(
        os.path.join(CHECKPOINT_DIR, "mobilenetv2_best.keras")
    )
    return model


def unfreeze_layers(model):
    #getting the base model last layer
    base_model = model.get_layer("mobilenetv2_1.00_224")

    base_model.trainable = True

    #freeze all layers
    for layer in base_model.layers :
        layer.trainable = False

    #unfreeze the last 30 layers
    for layer in base_model.layers[-30:] :
        layer.trainable = True
    return model, base_model


def create_finetuning_checkpoint():
    #recreating the checkpoint callback

    FINETUNING_DIR = "/home/user/Sync/backup/checkpoints"
    os.makedirs(FINETUNING_DIR, exist_ok=True)


    finetuning_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            FINETUNING_DIR, "mobilenetv2_finetuned_best.keras"
        ),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )
    return finetuning_checkpoint


def compile_model(model):
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate =0.0001 ),
        loss= 'categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model


def train_with_mlflow(model, train_dataset, val_dataset, finetuning_checkpoint, early_stopping_callback):
    #training with tracking with mlflow
    EPOCHS = 15
    BATCH_SIZE = 32

    with mlflow.start_run(run_name="MobileNetV2_Fine_Tuning"):

        # Track main parameters
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)

        start_time = time.time()

        history_finetuning = model.fit(
            train_dataset,
            validation_data = val_dataset,
            epochs = EPOCHS,
            callbacks = [
                finetuning_checkpoint,
                early_stopping_callback
            ]
        )

        # Track training duration
        mlflow.log_metric("training_time_seconds", time.time() - start_time)

        # Track best train results
        mlflow.log_metric(
                "best_train_accuracy",
                max(history_finetuning.history["accuracy"])
            )
        mlflow.log_metric(
                "best_train_loss",
                min(history_finetuning.history["loss"])
            )

            # Track best validation results
        mlflow.log_metric(
                "best_val_accuracy",
                max(history_finetuning.history["val_accuracy"])
            )
        mlflow.log_metric(
                "best_val_loss",
                min(history_finetuning.history["val_loss"])
            )

         # Save the trained model as an MLflow model
        mlflow.tensorflow.log_model(
            model=model,
            name="model",
            registered_model_name="NeuroVision_MobileNetV2_FT"
        )
    return history_finetuning


def save_history(history_finetuning):
    ARTIFACTS_DIR = "/home/user/Sync/backup/artifacts"
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    with open(os.path.join(ARTIFACTS_DIR,"fine_tuning_history.json"), "w") as f:
        json.dump(history_finetuning.history, f)
    return ARTIFACTS_DIR


def plot_finetuning_history(history_finetuning):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_finetuning.history["accuracy"], label="Training")
    plt.plot(history_finetuning.history["val_accuracy"], label="Validation")
    plt.title("Fine-tuning Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_finetuning.history["loss"], label="Training")
    plt.plot(history_finetuning.history["val_loss"], label="Validation")
    plt.title("Fine-tuning Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def load_histories(ARTIFACTS_DIR):
    # camparison tf vs ft

    with open(os.path.join(ARTIFACTS_DIR,"transfer_learning_history.json"), "r") as f:
        history_tl_data = json.load(f)

    with open(os.path.join(ARTIFACTS_DIR,"fine_tuning_history.json"), "r") as f:
        history_ft_data = json.load(f)
    return history_tl_data, history_ft_data


def plot_history(history, model_name):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Training")
    plt.plot(history["val_accuracy"], label="Validation")
    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Training")
    plt.plot(history["val_loss"], label="Validation")
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()


def set_champion():
    mlflow.set_tracking_uri(
        "https://billion-rides-keno-viewers.trycloudflare.com"
    )

    client = MlflowClient()

    MODEL_NAME = "NeuroVision_MobileNetV2_FT"
    MODEL_VERSION = "1"

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=MODEL_VERSION
    )

    print(f"Champion: {MODEL_NAME}, version {MODEL_VERSION}")
