import os
import pickle as pk
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


with open(os.path.join("data", "train_image_batches.pk"), "rb") as file:
    train_images = pk.load(file)
    
with open(os.path.join("data", "val_image_batches.pk"), "rb") as file:
    val_images = pk.dump(file)

# Use Tensorflow pretrained model
pretrained_model = tf.keras.applications.MobileNetV2(
input_shape= (224, 224, 3),
include_top = False,
weights = 'imagenet',
pooling = 'avg'
)

# Freeze weights
pretrained_model.trainable = False

# Create weights
inputs = pretrained_model.input

x = tf.keras.layers.Dense(128, activation = 'relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation = 'relu')(x)

outputs = tf.keras.layers.Dense(2, activation = 'softmax')(x)

model = tf.keras.Model(inputs = inputs, outputs = outputs)

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

# Train model
history = model.fit(
    train_images,
    validation_data = val_images,
    batch_size = 32,
    epochs = 20,
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 2,
            restore_best_weights = True
        )  
    ]
)