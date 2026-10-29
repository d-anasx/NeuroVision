import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def encode_labels(y):
    # Convert labels to ints and ints to categories
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    y_categorical  = to_categorical(y_encoded , num_classes=4)
    return label_encoder, y_categorical


def normalize(x):
    #normalization
    x = x.astype('float32') / 255.0
    return x


def create_dataset(x, y_categorical):
    #create tensor flow dataset
    dataset =  tf.data.Dataset.from_tensor_slices((x,y_categorical))
    #shuffling
    dataset = dataset.shuffle(
        buffer_size=len(x),
        seed=42,
        reshuffle_each_iteration=True
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE) # prepare the next batch of data while the model is training on the current batch.
    return dataset


def split_dataset(dataset):
    # instead of splitting data with the train_test_split we use the tensor flow method
    train_dataset, remaining_dataset = tf.keras.utils.split_dataset(
        dataset,
        left_size = 0.8,
        shuffle = True,
        seed= 42
    )

    val_dataset, test_dataset = tf.keras.utils.split_dataset(
        remaining_dataset,
        left_size = 0.5,
        shuffle = True,
        seed = 42
    )

    BATCH_SIZE = 32

    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_dataset, val_dataset, test_dataset


def create_data_augmentation():
    #data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ], name = 'data_augmentation')
    return data_augmentation
