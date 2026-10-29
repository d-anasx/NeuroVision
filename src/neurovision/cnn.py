import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model


def build_cnn_model(class_counts, data_augmentation):
    num_classes = len(class_counts)

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        data_augmentation,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.summary()
    return model


def compile_model(model):
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def create_callbacks(checkpoint_dir="/home/user/Sync/backup/checkpoints",
                     backup_dir="/home/user/Sync/backup/training_backup"):
    #Create the callbacks
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    backup_restore = tf.keras.callbacks.BackupAndRestore(
        backup_dir=backup_dir,
        save_freq="epoch"
    )

    #early stopping

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1

    )
    return model_checkpoint, backup_restore, early_stopping


def train_model(model, train_dataset, val_dataset, model_checkpoint, backup_restore, early_stopping):
    history = model.fit(
        train_dataset,
        validation_data = val_dataset,
        epochs = 20,
        callbacks = [model_checkpoint,
                     backup_restore,
                     early_stopping
                     ]
    )
    return history


def plot_history(history):
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['loss'], label = 'train loss')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.plot(history.history['val_loss'], label = 'val_loss')

    plt.xlabel('Epoch')
    plt.legend(loc='lower right')


def predict_sample(test_dataset, label_encoder):
    # prediction

    # getting an image from test data

    for images , labels in test_dataset.take(1):
        sample = images[0].numpy()
        true_label = label_encoder.classes_[tf.argmax(labels[0]).numpy()]


    plt.imshow(sample)
    plt.title(f"Actual class: {true_label}")
    plt.axis("off")
    plt.show()

    #getting the best model from checkpoint

    best_model = tf.keras.models.load_model(
        "/home/user/Sync/backup/checkpoints/best_model.keras"
    )

    #predicting
    probas = best_model.predict( np.expand_dims(sample, axis=0), verbose=0 )[0]

    predicted_index = np.argmax(probas)
    predicted_class = label_encoder.classes_[predicted_index]
    confidence = probas[predicted_index] * 100
    print("Actual class:", true_label)
    print("Predicted class:", predicted_class)
    print(f"Confidence: {confidence:.2f}%")
    return best_model


def save_architecture(best_model):
    plot_model(
        best_model,
        to_file="cnn_architecture.png",
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
        dpi=100
    )
