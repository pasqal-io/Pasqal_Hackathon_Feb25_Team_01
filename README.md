# Quantum AI for Multimodal Liver Cancer Prediction
## Team QScreen | The Blaise Pascal Quantum Challenge

A multimodal Quantum AI solution integrating clinical and imaging data in a Graph Neural Network (GNN) for early cancer detection. Enhances healthcare access, aligns with SDG 3, and transforms global health sustainability with real-world data integration.

## Overview of the project
### Problem statement


### Solution approach


## Installation and setup instructions
This project requires **Python 3.9**. The dependencies are listed in the _requirements.txt_ file and can be installed as follows:

```sh
pip install -r requirements.txt
```

## Data sources

Liver Cancer Dataset from The Cancer Imaging Archive: <a href="https://doi.org/10.7937/TCIA.5FNA-0924" target="_blank">HCC-TACE-Seg</a>

## Example use cases
Run embedding generation, training and testing from the _pipeline.py_ class. 
To generate clinical and image embeddings use mode _data_:

```sh
pipeline.py --mode data
```
Results will be saved in the _data_ directory and include embeddings both for training and testing.

Use modes _train_ and _test_ to train and test the model. Additionally, it is possible to cofigure paramters for each mode from the respective _config_ json files.

```sh
pipeline.py --config src/config/config_train.json --mode train
```

```sh
pipeline.py --config src/config/config_test.json --mode test
```

