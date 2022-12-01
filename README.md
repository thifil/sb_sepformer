# Reimplementation of "Attention is all you need in Speech Seperation"
In this work I have reimplemented the paper "Attention is all you need in Speech Speration", for details please view https://ieeexplore.ieee.org/abstract/document/9413901. 
## Paper Abstract
This paper is adopting the transformer architecture from the paper "Attention is all you need" into the speech domain, aiming to seperate multiple speakers. Input to this model is a single channel audio stream with 2-3 speaker speaking at the same time. The model then generates a mask for each speaker and, thus, seperates each speaker into a distinct output stream. 
The advantage of using Transformers over RNNs is that the processing of time based signals can be done in a parallelized manner, instead of sequentially. A special characteristic of SepFormer is that it learns short and long-term dependencies utilizing a positional embedding across the window and across the chunks. 

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
1. Change the "homepath" to the directory you are going to execute the code in.
2. verify "data_folder" points to the data directory to be used, and the mixture.csv files are present.

### Executing a run 
run a training by executing.

```python sepformer.py```

### Changing Hyperparamters
To alter the training parameters open ```config_sepformer.yaml```.
Here you can change the filepaths, the model parameters, experiment parameters loss function and optimizer.