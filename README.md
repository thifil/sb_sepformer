# Reimplementation of "Attention is all you need in Speech Seperation"
## Setup

### Data Generation
For this implementation Libri2Mix was used as a dataset, with 360 hours of clean speech for training.
To obtain the dataset follow the guide in this repository: https://github.com/JorisCos/LibriMix
To properly generate the data you have to execute the script as is, but you can subsequently remove the wav16k folder as only wav8k data is used.

### Dependencies
The dependencies required to run sepformer.py are listed in dependencies.txt. Please install the packages according to this list. Make sure the data folder is in the same folder as sepformer.py and config_sepformer.yaml.

### Experiment tracking

To track experiment performance, you need to integrate wandb.ai. Follow their quickstart guide for setup instructions https://docs.wandb.ai/quickstart. 

## Run
### Configuration of config_sepformer.yaml
1. Change the "hompath" to the directory you are going to execute the code in.
2. verify "data_folder" points to the data directory to be used, and the mixture.csv files are present.

### Executing a run
run a training by executing.

_python sepformer.py_