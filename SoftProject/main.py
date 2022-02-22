import os

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def prepare_single_image(img_path: str) -> np.array:
    img = Image.open(img_path)
    img = img.resize(size=(180, 180))
    return np.array(img) / 255.0


def display_image(image):
    plt.imshow(image, 'gray')


def loadnormal():
    train_dir = 'test\\normal'
    loadimage(train_dir)


def loadopacity():
    train_dir = 'test\\covid'
    loadimage(train_dir)


def loadimage(train_dir):
    for directory, subdirectories, files in os.walk(train_dir):
        for file in files:
            path = os.path.join(directory, file)
            img = load_image(path)
            display_image(img)
            plt.show()


def defineTrainingImages(pathFolder):
    data_dir = 'train'
    validation_data = 'val'
    batch_size = 64
    img_height = 150
    img_width = 150

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_data,
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomTranslation(height_factor=0.2, width_factor = 0.2)
    ])

    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        data_augmentation,
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs = 20
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    #train_datagen_augmented = ImageDataGenerator

    num_total_cat, num_correct_cat = 0, 0
    num_total_dog, num_correct_dog = 0, 0



    for directory, subdirectories, files in os.walk(pathFolder):
        for file in files:
            path = os.path.join(directory, file)
            img = tf.keras.utils.load_img(
                path, target_size=(img_height, img_width)
            )
            i += 1
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = model.predict(img_array)
            pred = predictions.argmax()
            if pred == 1:
                num_correct_dog += 1
            else:
                num_total_cat += 1
            score = tf.nn.softmax(predictions[0])


            print(
                "{}. image most likely belongs to {} with a {:.2f} percent confidence."
                    .format(i,class_names[np.argmax(score)], 100 * np.max(score))
            )

        print(num_correct_dog)
        print(num_correct_cat)


def loadnormal():
    pathFolder = 'test\\normal'
    defineTrainingImages(pathFolder)


def loadcovid():
    pathFolder = 'test\\covid'
    defineTrainingImages(pathFolder)


loadcovid()
loadnormal()