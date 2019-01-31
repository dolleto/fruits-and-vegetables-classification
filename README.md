# fruits-and-vegetables-classification

Authors: Louis De Bellevue, Axelle Albot, Thibaud Lhermitte, Nikolay Ionanov, Alexandre Dollé

## Getting Started

A [dedicated notebook](fv_classification_starting_kit.ipynb)
is available to get you started on the problem.

### Downloading data

Download the data (~5GB) by running

```
python download_data.py
```

the first time. It will download a zip file and unzip it in your current project folder.

### Installing dependencies

The installation script [`install.sh`](install.sh) used to make the AMI
is also available. Depending on your current installation, you may not need
to execute all of this, but it shows the versions of the various libraries
against which we tested the starting kit.

### Keras channel

You should set `image_data_format` to `channels_last` in `~/.keras/keras.json`.
