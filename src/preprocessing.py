from src import load_data
import tensorflow as tf
import pickle as pk
import os


train_image_paths, valid_image_paths = load_data.data_path_loader("data")

# Create a train and validation DataFrame
train_df = load_data.image_processing(train_image_paths)
val_df = load_data.image_processing(valid_image_paths)

# Generate new images from dataset
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
)

val_generator = tf.keras.preprocessing.image.ImageDataGenerator(
preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
)

# Generate images using 'train_df' DataFrame
train_images = train_generator.flow_from_dataframe(
    dataframe  = train_df,
    x_col = 'Filepath',
    y_col = 'Label',
    target_size = (224, 224),
    color_mode = 'rgb',
    class_mode = 'categorical',
    batch_size = 32,
    shuffle = True,
    seed = 0,
    rotation_range = 30,
    zoom_range = 0.15,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    horizontal_flip = True,
    fill_mode = 'nearest')

# Generate images using 'val_df' DataFrame
val_images = train_generator.flow_from_dataframe(
    dataframe  = val_df,
    x_col = 'Filepath',
    y_col = 'Label',
    target_size = (224, 224),
    color_mode = 'rgb',
    class_mode = 'categorical',
    batch_size = 32,
    shuffle = True,
    seed = 0,
    rotation_range = 30,
    zoom_range = 0.15,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    horizontal_flip = True,
    fill_mode = 'nearest'
)


with open(os.path.join("data", "train_image_batches.pk"), "wb") as file:
    pk.dump(train_images, file)
    
with open(os.path.join("data", "val_image_batches.pk"), "wb") as file:
    pk.dump(val_images, file)