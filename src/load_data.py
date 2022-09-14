# Import necessary libraries
import pandas as pd
import pathlib
import os
import subprocess
from zipfile import ZipFile
import yaml
from argparse import ArgumentParser
from src.logs import get_logger
from typing import List, Tuple





parser = ArgumentParser()
parser.add_argument("--path", "--p", default="params.yaml", dest="path", type=str, required=True)
args = parser.parse_args()
param_path = args.path

with open(param_path) as file:
    config = yaml.safe_load(file)


logger = get_logger('Data Loading', log_level=config['loglevel'])


logger.info("Checking if the zip file path exist")
if os.path.exists(config["paths"]["zip_data"]):
    logger.info("path exist")
    if (os.path.exists(os.path.join(config["paths"]["data_root"], config["paths"]["train_path"])) & os.path.exists(os.path.join(config["paths"]["data_root"], config["paths"]["val_path"]))):
        logger.info("Train and Val set has been extracted")
    else:
        logger.info("Extracting the zip file into data directory")
        with ZipFile(config["paths"]["zip_data"], "r") as zip:
            zip.extractall("data/")
        
else:
    logger.info("download the gender-classification-dataset")
    result = subprocess.Popen("kaggle datasets download -d cashutosh/gender-classification-dataset", shell=True, stdout=subprocess.PIPE)
    logger.info(result.communicate()[0].decode("utf-8"))
    logger.info("Extracting the zip file into data directory")
    with ZipFile(config["paths"]["zip_data"], "r") as zip:
            zip.extractall("data/")
            


def data_path_loader(rootpath:str) -> Tuple[List[str], List[str]]:
    logger.info('functioning running to load the path of images and there corresponding labels')
    # Get directory path to Training dataset
    train_dir = pathlib.Path(os.path.join(rootpath, config["paths"]["train_path"]))
    # Get a list of all images in the Training dataset
    train_image_paths = list(train_dir.glob(r'**/*.jpg'))

    # Get directory path to Validation dataset
    valid_dir = pathlib.Path(os.path.join(rootpath, config["paths"]["val_path"]))
    # Get a list of all images in the Validation dataset
    valid_image_paths = list(valid_dir.glob(r'**/*.jpg'))
    return train_image_paths, valid_image_paths

def image_processing(filepath) -> pd.DataFrame:
    logger.info('functioning converting the path of images and the labels into a dataframe')
    labels = [str(filepath[i]).split('/')[-2]
             for i in range(len(filepath))]
    
    # Create a DataFrame and input the filepath and labels
    filepath = pd.Series(filepath, name = 'Filepath').astype(str)
    labels = pd.Series(labels, name = 'Label')
    
    df = pd.concat([filepath, labels], axis='columns')
    
    return df






