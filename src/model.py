from datetime import datetime
import yaml
from tensorflow import keras
from argparse import ArgumentParser
from logs import get_logger
from preprocessing import train_images, val_images
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from tensorflow import keras




parser = ArgumentParser()
parser.add_argument("--path", "--p", default="params.yaml", dest="path", type=str, required=True)
args = parser.parse_args()
param_path = args.path

with open(param_path) as file:
    config = yaml.safe_load(file)

logger = get_logger('Training', log_level=config['loglevel'])



# Use Tensorflow pretrained model
logger.info("loading the val the pretrain model")
pretrained_model = keras.applications.MobileNetV2(
input_shape= ((config["preprocessing"]["target_size"], ) * 2) + (3, ),
include_top = False,
weights = config["training"]["pretrain_w"],
pooling = config["training"]["pretrain_p"]
)

# Freeze weights
pretrained_model.trainable = False

# Create weights
inputs = pretrained_model.input

x = keras.layers.Dense(128, activation = 'relu')(pretrained_model.output)
x = keras.layers.Dense(128, activation = 'relu')(x)

outputs = keras.layers.Dense(2, activation = 'softmax')(x)

model = keras.Model(inputs = inputs, outputs = outputs)

model.compile(
    optimizer = config["training"]["opt"],
    loss = config["training"]["loss"],
    metrics = [config["training"]["metrics"]]
)

logger.info(model.summary())

earlystop = EarlyStopping(
            monitor = 'val_loss',
            patience = 2,
            restore_best_weights = True)  
model_names = "model/checkpoint/model.{epoch:02d}-{val_accuracy:.2f}.hdf5" 
checkpoint = ModelCheckpoint(model_names, monitor="val_accuracy", verbose=1, save_best_only=True, save_weights_only=False)
log_file_path = "model/log.txt"
csv_logger = CSVLogger(log_file_path, append=False)
tensorboard_log_dir = "model/logs_" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)


# Train model
logger.info("Training the data")
history = model.fit(
    train_images,
    validation_data = val_images,
    batch_size = 32,
    epochs = 100,
    callbacks = [earlystop, csv_logger, checkpoint, tensorboard]
)


logger.info("saving the model")
model.save(config["paths"]["model"])