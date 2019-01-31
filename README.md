# fruits-and-vegetables-classification

Authors: Martin Regnaud, Louis De Bellevue, Axelle Albot, Thibaud Lhermitte, Nikolay Ionanov, Alexandre Dollé

This challenge was created in the context of an academic course of the M2 Data Science.

## Getting Started

A [dedicated notebook](fv_classification_starting_kit.ipynb)
is available to get you started on the problem.

### Downloading data

Download the data (~5GB) by running

```
python download_data.py
```

the first time. It will download a zip file and unzip it in your current project folder.


### Troubleshooting 

The ramp_test_submission doesn't work for the moment : the first cross-validation is launched but the training never ends. We believe it is linked to the way images are treated within the ramp workflow. The training is properly laucnhed but it gets stuck on an image. We went through all possible ideas but couldn't spot which type of images cause problems. 

When we train the starting_kit locally (without the ramp frame) the training ends (and works). We obtained an accuracy of 50% when testing the model. You can rerun our local training by runing the following steps :
- download the data using [this link](https://www.dropbox.com/s/cpw276pvtcgtc65/dataset_v2_train_test.zip?dl=0) : https://www.dropbox.com/s/cpw276pvtcgtc65/dataset_v2_train_test.zip?dl=0
- unzip the folder and put it in the root of the project
- run the jupyter notebook called [local_baseline_model.ipynb](local_baseline_model.ipynb)

### utils

The utils directory contains the scripts that allowed us to fetch images from bing and flickr. We used these scripts to collect images corresponding to certain key words (names of the fruits and vegetables). We then filtered the images manually and cceated our dataset with the remaining images.

### Installing dependencies

The installation script [`install.sh`](install.sh) used to make the AMI
is also available. Depending on your current installation, you may not need
to execute all of this, but it shows the versions of the various libraries
against which we tested the starting kit.
