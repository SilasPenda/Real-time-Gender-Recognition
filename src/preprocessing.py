import load_data
import tensorflow as tf
import pickle as pk
import os
import yaml
from argparse import ArgumentParser
from logs import get_logger

parser = ArgumentParser()
parser.add_argument("--path", "--p", default="params.yaml", dest="path", type=str, required=True)
args = parser.parse_args()
param_path = args.path


with open(param_path) as file:
    config = yaml.safe_load(file)

logger = get_logger("Preprocessing", log_level=config['loglevel'])

train_image_paths, valid_image_paths = load_data.data_path_loader(config["paths"]["data_root"])
logger.info("The paths of the images and the labels for the train and val set has been loaded")

# Create a train and validation DataFrame
train_df = load_data.image_processing(train_image_paths)
logger.info("The paths of the images and the labels for the train set has been transfromed into a DataFrame format")
val_df = load_data.image_processing(valid_image_paths)
logger.info("The paths of the images and the labels for the val set has been transfromed into a DataFrame format")


# Generate new images from dataset
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
)

val_generator = tf.keras.preprocessing.image.ImageDataGenerator(
preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
)

# Generate images using 'train_df' DataFrame
logger.info("Reading train set into tensorflow data format")
train_images = train_generator.flow_from_dataframe(
    dataframe  = train_df,
    x_col = config["preprocessing"]["x_col"],
    y_col = config["preprocessing"]["y_col"],
    target_size = (config["preprocessing"]["target_size"], ) * 2,
    color_mode = config["preprocessing"]["color_mode"],
    class_mode = config["preprocessing"]["class_mode"],
    batch_size = config["base"]["batch_size"],
    shuffle = config["preprocessing"]["shuffle"],
    seed = config["preprocessing"]["seed"],
    rotation_range = config["preprocessing"]["rotation_range"],
    zoom_range = config["preprocessing"]["zoom_range"],
    width_shift_range = config["preprocessing"]["width_shift_range"],
    height_shift_range = config["preprocessing"]["height_shift_range"],
    shear_range = config["preprocessing"]["shear_range"],
    horizontal_flip = config["preprocessing"]["horizontal_flip"],
    fill_mode = config["preprocessing"]["fill_mode"])
logger.info("sucessfully")



# Generate images using 'val_df' DataFrame
logger.info("Reading val set into tensorflow data format")
val_images = train_generator.flow_from_dataframe(
    dataframe  = val_df,
    x_col = config["preprocessing"]["x_col"],
    y_col = config["preprocessing"]["y_col"],
    target_size = (config["preprocessing"]["target_size"], ) * 2,
    color_mode = config["preprocessing"]["color_mode"],
    class_mode = config["preprocessing"]["class_mode"],
    batch_size = config["base"]["batch_size"],
    shuffle = config["preprocessing"]["shuffle"],
    seed = config["preprocessing"]["seed"],
    rotation_range = config["preprocessing"]["rotation_range"],
    zoom_range = config["preprocessing"]["zoom_range"],
    width_shift_range = config["preprocessing"]["width_shift_range"],
    height_shift_range = config["preprocessing"]["height_shift_range"],
    shear_range = config["preprocessing"]["shear_range"],
    horizontal_flip = config["preprocessing"]["horizontal_flip"],
    fill_mode = config["preprocessing"]["fill_mode"]
)

logger.info("sucessfully")