import os
import pathlib
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image, ImageOps
from sklearn.utils import shuffle
from matplotlib import rcParams
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from numpy.random import seed
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False


def process_image(img_path: str) -> np.array:
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    img = img.resize(size=(150, 150))
    img = np.ravel(img) / 255.0
    return img


def process_folder(folder: pathlib.PosixPath) -> pd.DataFrame:
    # We'll store the images here
    processed = []

    # For every image in the directory
    for img in folder.iterdir():
        # Ensure JPG
        if img.suffix == '.jpeg':
            # Two images failed for whatever reason, so let's just ignore them
            try:
                processed.append(process_image(img_path=str(img)))
            except Exception as _:
                continue

    # Convert to pd.DataFrame
    processed = pd.DataFrame(processed)
    # Add a class column - dog or a cat
    processed['class'] = folder.parts[-1]

    return processed


def defineTrainingImages():
    seed(0)
    tf.random.set_seed(0)
    # Training set
    train_covid = process_folder(folder=pathlib.Path.cwd().joinpath('train/covid'))
    train_normal = process_folder(folder=pathlib.Path.cwd().joinpath('train/normal'))
    train_set = pd.concat([train_covid, train_normal], axis=0)

    # Test set
    test_covid = process_folder(folder=pathlib.Path.cwd().joinpath('test/covid'))
    test_normal = process_folder(folder=pathlib.Path.cwd().joinpath('test/normal'))
    test_set = pd.concat([test_covid, test_normal], axis=0)

    # Validation set
    valid_covid = process_folder(folder=pathlib.Path.cwd().joinpath('val/covid'))
    valid_normal = process_folder(folder=pathlib.Path.cwd().joinpath('val/normal'))
    valid_set = pd.concat([valid_covid, valid_normal], axis=0)

    train_set.head()

    train_set = shuffle(train_set).reset_index(drop=True)
    valid_set = shuffle(valid_set).reset_index(drop=True)

    X_train = train_set.drop('class', axis=1)
    y_train = train_set['class']

    X_valid = valid_set.drop('class', axis=1)
    y_valid = valid_set['class']

    X_test = test_set.drop('class', axis=1)
    y_test = test_set['class']
    y_train.factorize()

    y_train = tf.keras.utils.to_categorical(y_train.factorize()[0], num_classes=2)
    y_valid = tf.keras.utils.to_categorical(y_valid.factorize()[0], num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test.factorize()[0], num_classes=2)
    class_names = ['covid', 'normal']
    print(class_names)
    X_train_array = X_train.to_numpy()
    datagen = ImageDataGenerator(rotation_range=90)
    datagen.fit(X_train_array.reshape(-1,150,150,1))
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax'),
    ])

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    callbacks =[
        EarlyStopping(patience=5)
    ]
    history = model.fit(

        X_train,
        y_train,
        epochs=100, #100
        batch_size=10, #10
        validation_data=(X_valid, y_valid),
       # callbacks=callbacks
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(100)

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

    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])

    predictions = probability_model.predict(X_test, batch_size=128, verbose=0)
    for i in range(len(X_test)):
        print(
            "{}. image most likely belongs to {} with a {:.5f} percent confidence."

                .format(i + 1, class_names[np.argmax(predictions[i])], 100 * np.max(predictions[i]))
        )

    num_covid, num_correct_covid = 0, 0
    num_normal, num_correct_normal = 0, 0
    num_correct_normal2, num_correct_covid2 = 0, 0
    for img_path in pathlib.Path.cwd().joinpath('test/covid').iterdir():
        try:
            img = process_image(img_path=str(img_path))
            pred = model.predict(tf.expand_dims(img, axis=0))
            pred = pred.argmax()
            num_covid += 1
            if pred == 0:
                num_correct_covid += 1
            else:
                num_correct_normal += 1

        except Exception as e:
            continue
    print("Amount of patients with covid:", num_covid)
    print("Amount of wrong covid classification:", num_correct_normal)
    print("Amount of correct covid classification:", num_correct_covid)
    print("Accuracy: {:.2f} %".format((num_correct_covid / num_covid)*100))
    for img_path in pathlib.Path.cwd().joinpath('test/normal').iterdir():
        try:
            img = process_image(img_path=str(img_path))
            pred = model.predict(tf.expand_dims(img, axis=0))
            pred = pred.argmax()
            num_normal += 1
            if pred == 1:
                num_correct_normal2 += 1
            else:
                num_correct_covid2 += 1
        except Exception as e:
            continue
    print("Amount of patients with normal lungs:", num_normal)
    print("Amount of correct normal predictions:", num_correct_normal2)
    print("Amount of wrong normal predictions:", num_correct_covid2)
    print("Accuracy: {:.2f} %".format((num_correct_normal2 / num_normal)*100))


defineTrainingImages()
