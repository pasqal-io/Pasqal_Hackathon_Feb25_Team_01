# Quantum AI for Multimodal Liver Cancer Classification
## Team QScreen | The Blaise Pascal Quantum Challenge

A multimodal Quantum AI solution integrating clinical and imaging data in a Graph Neural Network (GNN) for classifying cancer progression. Enhances healthcare access, aligns with SDG 3, and transforms global health sustainability with real-world data integration.

## Overview of the project
### Project Structure
Below is an overview of the project structure. 

```sh
.
 |-test
 | |-clinical-classification.ipynb
 | |-mwe.ipynb
 |-docs
 |-results
 |-data
 | |-clinical_data
 | | |-clinical_Data.csv
 | | |-train_embeddings.npy
 | | |-test_embeddings.npy
 | |-labels
 | | |-train_labels.csv
 | | |-test_labels.csv
 | |-image_data
 | | |-png
 | | |-train_embeddings.npy
 | | |-test_embeddings.npy
 | |-test_data.csv
 | |-train_data.csv
 | |-data_loader.py
 |-src
 | |-pvem
 | | |-clinical_data_embeddings.ipynb
 | | |-clinical_data_embeddings.py
 | |-qcnn
 | | |-quanvolution.py
 | |-rmrm
 | | |-utils
 | | | |-utils.py
 | | |-embeddings.py
 | | |-model
 | | | |-intialisation.py
 | | | |-network.py
 | | | |-components.py
 | |-config
 | | |-config_train.json
 | | |-config_test.json
 | |-test.py
 | |-train.py
 |-pipeline.py
 |-requirements.txt
```

The network implementation can be found inside of the _src_ folder where: 
- **pvem**: performs clinical preprocessing and generates clinical embeddings
- **qcc**: performs the quantum convolutions and generates image embeddings
- **rmrm**: combines the previous embeddings in a graph neural network and performs classification
- **config**: contains configuration files used at train and test time

Additionally, _data_ folder contains: 
- **clinical_data**: the CSV data used at train and test time to generate clinical embeddings and the corresponding target labels 
- **image_data**: a folder containing a png imge for each patient (name format: x.png where x is the id of the patient)
After running the generation of the embeddings, the embeddings will be stored as .npy files in the respective folder.

The whole pipeline is handled by _pipeline.py_ in the root of the project. Refer to the following section for some example use cases.

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

