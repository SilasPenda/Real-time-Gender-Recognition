# Import necessary libraries
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import PIL.Image as Image
import pathlib
import os
def data_path_loader(rootpath:str):
    # Get directory path to Training dataset
    train_dir = pathlib.Path(os.path.join(rootpath, 'Training'))
    # Get a list of all images in the Training dataset
    train_image_paths = list(train_dir.glob(r'**/*.jpg'))

    # Get directory path to Validation dataset
    valid_dir = pathlib.Path(os.path.join(rootpath, 'Validation'))
    # Get a list of all images in the Validation dataset
    valid_image_paths = list(valid_dir.glob(r'**/*.jpg'))
    return train_image_paths, valid_image_paths

def image_processing(filepath):
    labels = [str(filepath[i]).split('/')[-2]
             for i in range(len(filepath))]
    
    # Create a DataFrame and input the filepath and labels
    filepath = pd.Series(filepath, name = 'Filepath').astype(str)
    labels = pd.Series(labels, name = 'Label')
    
    df = pd.concat([filepath, labels], axis='columns')
    
    return df






