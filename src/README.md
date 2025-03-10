# Source Code Directory: Quantum-Enhanced Cancer Survival Prediction

This directory contains the core implementation of our Quantum AI solution for early cancer detection and patient survival prediction, developed for the Pasqal Hackathon.

## Architecture Overview

Our solution uses a hybrid quantum-classical approach for multimodal cancer data analysis:

1. **Quantum CNN (QCNN)** - Processes medical images using quantum computing techniques
2. **Patient Variable Embedding Model (PVEM)** - Processes clinical data using classical deep learning
3. **Region-based Multimodal Relational Module (RMRM)** - Combines both modalities using graph neural networks

## Directory Structure

### Main Scripts

- `train.py` - The main training script for the complete multimodal model
- `test.py` - Evaluation script to test model performance with various metrics (C-index, accuracy, etc.)

### Core Components

- `rmrm/` - Region-based Multimodal Relational Module implementation
  - `model/` - Neural network architecture components
    - `network.py` - Main network combining image and clinical embeddings
    - `components.py` - RMRM implementation using Graph Attention Networks
    - `intialisation.py` - Weight initialization utilities
  - `embeddings.py` - Dataset handling for precomputed embeddings
  - `utils/` - Utility functions for data processing and evaluation

- `qcnn/` - Quantum Convolutional Neural Network implementation
  - `quanvolution.py` - Quantum convolution operations using Qadence framework
  - `qmain.py` - Training and embedding generation for the QCNN

- `pvem/` - Patient Variable Embedding Model for clinical data
  - `clinical_data_embeddings.py` - Feature extraction and embedding generation for clinical records

## Data Flow

1. Medical images are processed through the QCNN to generate quantum-enhanced embeddings
2. Clinical data is processed through PVEM to generate patient-level embeddings
3. Both embeddings are combined in the RMRM using graph neural networks
4. The final output is a binary classification indicating patient survival probability

## Dependencies

- PyTorch and PyTorch Geometric for neural networks and graph operations
- Qadence for quantum computing components
- Scikit-learn for evaluation metrics and classical ML components
- Pandas and NumPy for data processing