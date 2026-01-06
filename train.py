import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import load_img
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions
from tensorflow.keras import layers, models


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                  rotation_range=20,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

train_data = train_datagen.flow_from_directory(
    "/path/to/train",
    target_size=(299, 299),
    batch_size=32)


val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_data = val_datagen.flow_from_directory(
    "/path/to/val",
    target_size=(299, 299),
    batch_size=32,
    shuffle=False
)

def make_model(learning_rate, inner_size, droprate):
  base_model = Xception(weights="imagenet", input_shape=(299, 299, 3), include_top=False)

  base_model.trainable = False

  model = models.Sequential([
    base_model,

    layers.GlobalAveragePooling2D(),
    layers.Dense(inner_size, activation='relu'),
    layers.Dropout(droprate),
    layers.Dense(15)
  ])

  optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
  loss = keras.losses.CategoricalCrossentropy(from_logits=True)
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  return model

checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

model = make_model(learning_rate=0.001, inner_size=100, droprate=0.2)
model.fit(train_data, epochs=10, validation_data=val_data, callbacks=[checkpoint])