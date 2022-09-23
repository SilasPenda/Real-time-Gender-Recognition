# Real-time-Gender-Detection

# INTRODUCTION
Real time gender detection has become a critical component in the new domain of computer human observation and Computer Human Interaction (HCI). Gender detection has numerous applications in the area of recommender systems, focused advertising, security and surveillance. Detection of gender by using the facial features is done by many methods such as Gabor wavelets, artificial neural networks and support vector machine.
In this project, we have used deep learning as a pre-cursor.
<br>

# DATA
<!-- you should add the link to download data below-->
- Dataset can be downloaded from [Kaggle](link) and placed in the root directory.
<!-- I dont understand this line we didn't download any black men and women images from chrome -->
- Used chrome extension to download images of black men and women with afro.
- The dataset for the training can also be downloaded by running the notebook or by using the dvc pipeline. This require setting up kaggle api configuration detailed instruction can be find [below]()



# TOOLS
Below are the list of tools used in building the gender detection model

* Python
* Tensorflow
* OpenCV
* cvlib
* DVC
* Matplotlib
* Pandas
* Numpy

# SET-UP

You can set up an environment for the gender detection model by using the following commands:

- Step one is creating a virtual environment in the root directory
`$ python -m venv gender_detection` <br>
- The path to source files should be added to the <b> PYTHONPATH </b>
- All dependencies can be installs by running the line below 
`$ pip install -r requirement.txt` <br>

## DVC for MLOPs

Alongside git DVC was used in the project to do data version control and to build a pipeline automation and configuration for the project.
`model pipeline` 
<img src="report\model_pipeline.png">
```
ABOUT DVC

DVC is built upon Git and its main goal is to codify data, models and piplines through the command line. Although DVC can work stand-alone, it's highly recommended to work aloginside <strong>Git</strong>

DVC can be installed as a Python Library with pip package manager:

`$ pip install dvc` <br>

It is also possible to isntall DVC using conda:<br>
`$ conda install -c conda-forge mamba`<br>
`$ mamba install -c conda-forge dvc`<br>
`$ mamba install -c conda-forge dvc-s3`<br>

After installing DVC, you can go to your project's folder and initialize both Git and DVC: <br>
`$ git init`<br>
`$ dvc init` 
for more visit https://www.dvc.org/
```
## Training
The train set was trained using imagenet pretrained mobilenetv2 levaraging on the pretrain model
# PERFORMANCE
The model performance was track by using training and validation accuracies and losses as shown in the image below
## Classification Report
`Accuracy`
<img src="report\epoch_accuracy.JPG"> 
`Loss`
<img src="report\epoch_loss.JPG">

# RECOMMENDATIONS
# LINKS