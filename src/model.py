import os
import yaml
import pickle as pk
import tensorflow as tf
import tensorflow_hub as hub
from argparse import ArgumentParser
from src.logs import get_logger

from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential



parser = ArgumentParser()
parser.add_argument("--path", "--p", default="params.yaml", dest="path", type=str, required=True)
args = parser.parse_args()
param_path = args.path

with open(param_path) as file:
    config = yaml.safe_load(file)

logger = get_logger('Training', log_level=config['loglevel'])


logger.info("loading the train batch")
with open(os.path.join(config["paths"]["data_root"], config["paths"]["processed_train_path"]), "rb") as file:
    train_images = pk.load(file)
    
logger.info("loading the val batch")
with open(os.path.join(config["paths"]["data_root"], config["paths"]["processed_val_path"]), "rb") as file:
    val_images = pk.dump(file)

# Use Tensorflow pretrained model
logger.info("loading the val the pretrain model")
pretrained_model = tf.keras.applications.MobileNetV2(
input_shape= ((config["preprocessing"]["target_size"], ) * 2) + (3, ),
include_top = False,
weights = config["training"]["pretrain_w"],
pooling = config["training"]["pretrain_p"]
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
    optimizer = config["training"]["opt"],
    loss = config["training"]["loss"],
    metrics = [config["training"]["metrics"]]
)

logger.info(model.summary())

# Train model
logger.info("Training the data")
history = model.fit(
    train_images,
    validation_data = val_images,
    batch_size = config["base"]["batch_size"],
    epochs = config["training"]["epoch"],
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor = config["training"]["monitor"],
            patience = 2,
            restore_best_weights = True
        )  
    ]
)


logger.info("saving the model")
model.save(config["paths"]["model"])