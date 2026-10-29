import os
import time
import json
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2


def setup_mlflow():
    # ml flow setup
    mlflow.set_tracking_uri("https://billion-rides-keno-viewers.trycloudflare.com")

    mlflow.set_experiment("NeuroVision_AI")

    experiment = mlflow.get_experiment_by_name("NeuroVision_AI")
    return experiment


def create_checkpoint_dir():
    # checkpoints folder
    CHECKPOINT_DIR = "/home/user/Sync/backup/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return CHECKPOINT_DIR


# Model Architecture

def create_tumor_classifier():

    # define the input layer
    inputs = layers.Input(
        shape = (224,224,3),
        name = "mri_images"
    )

    # rescaling the images to -1 , 1 for MobileNetV2 (we can use process_input())
    # x new = 2 * x old - 1
    x = layers.Rescaling(
        scale = 2.0,
        offset = -1.0,
        name = 'mobilenet_preprocessing'
    )(inputs)

    #loading the pretrained MobileNetV2
    base_model = MobileNetV2( weights ='imagenet', include_top = False, input_shape =(224, 224, 3))

    # freeze the pretrained layers
    base_model.trainable = False

    # extract images features
    x = base_model(x, training=False)


    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(4, activation='softmax', name='Classification')(x)

    # Assemble the model

    model = Model(
        inputs = inputs,
        outputs = outputs
    )

    return model, base_model


def compile_model(model):
    # compile the model
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate =0.001 ), # Adam: updates the trainable weights.
        loss= 'categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model


def create_callbacks(CHECKPOINT_DIR):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, 'mobilenetv2_best.keras'),
        monitor = 'val_loss',
        save_best_only = True,
        mode = 'min',
        verbose = 1
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor = "val_loss",
        patience = 3,
        restore_best_weights = True,
        verbose = 1
    )
    return checkpoint_callback, early_stopping_callback


def train_with_mlflow(model, train_dataset, val_dataset, checkpoint_callback, early_stopping_callback):
    #training with tracking with mlflow
    EPOCHS = 15
    BATCH_SIZE = 32

    with mlflow.start_run(run_name="MobileNetV2_Transfer_Learning"):

        # Track main parameters
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)

        start_time = time.time()

        history = model.fit(
            train_dataset,
            validation_data = val_dataset,
            epochs = EPOCHS,
            callbacks = [
                checkpoint_callback,
                early_stopping_callback
            ]
        )

        # Track training duration
        mlflow.log_metric("training_time_seconds", time.time() - start_time)

        # Track best train results
        mlflow.log_metric(
                "best_train_accuracy",
                max(history.history["accuracy"])
            )
        mlflow.log_metric(
                "best_train_loss",
                min(history.history["loss"])
            )

            # Track best validation results
        mlflow.log_metric(
                "best_val_accuracy",
                max(history.history["val_accuracy"])
            )
        mlflow.log_metric(
                "best_val_loss",
                min(history.history["val_loss"])
            )

         # Save the trained model as an MLflow model
        mlflow.tensorflow.log_model(
            model=model,
            name="model",
            registered_model_name="NeuroVision_MobileNetV2"
        )
    return history


def save_history(history):
    ARTIFACTS_DIR = "/home/user/Sync/backup/artifacts"
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    with open(os.path.join(ARTIFACTS_DIR,"transfer_learning_history.json"), "w") as f:
        json.dump(history.history, f)
    return ARTIFACTS_DIR


def plot_history(history):
    # plotting the history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
