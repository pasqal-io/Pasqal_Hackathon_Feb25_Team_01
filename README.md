# ⚛️ Quantum AI for Multimodal Early Cancer Detection 🔬🏥
## The Blaise Pascal Quantum Challenge

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multimodal Quantum AI solution integrating clinical and imaging data in a Graph Neural Network (GNN) for early cancer detection. Enhances healthcare access, aligns with SDG 3, and transforms global health sustainability with real-world data integration.

## Overview of the project

This project presents an innovative approach to early cancer detection by combining multimodal data sources - clinical information and medical imaging - through a graph-based deep learning architecture. By representing patients as graphs with interconnected clinical and imaging nodes, our model captures complex relationships between different data modalities, improving detection accuracy and reliability.

### Problem statement

Early cancer detection remains challenging due to:
- The siloed nature of medical data (clinical records separate from imaging)
- Difficulty in identifying subtle patterns across different data sources
- Need for improved sensitivity without sacrificing specificity
- Limited access to advanced diagnostic tools in resource-constrained settings

Our solution addresses these challenges through multimodal integration of patient data, enhancing detection capabilities while maintaining explainability and accessibility.

### Solution approach

We developed a Graph Neural Network (GNN) architecture that:

1. **Represents patients as graphs**: Each patient is modeled as a graph with clinical and imaging nodes
2. **Connects multimodal data**: Creates edges between clinical features and imaging features
3. **Leverages graph convolutions**: Uses graph convolutional networks (GCN) to process interconnected data
4. **Implements feature attention**: Employs attention mechanisms to weigh the importance of different modalities
5. **Provides binary classification**: Outputs cancer/no-cancer prediction with high accuracy

Our model variants include:
- Standard GCN with multimodal integration
- Graph Attention Networks (GAT) for weighted feature processing
- Feature attention to dynamically balance clinical vs. imaging importance

## Installation and setup instructions

This project requires **Python 3.9**. The dependencies are listed in the _requirements.txt_ file and can be installed as follows:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Data sources

The model uses two main data sources:

1. **Clinical data**: 
   - Patient metrics and clinical features
   - 38 clinical nodes per patient
   - Embeddings of dimension 128

2. **Medical imaging**:
   - Processed imaging data (likely radiological scans)
   - 36 image nodes per patient (6x6 grid)
   - Embeddings of dimension 128

The dataset consists of:
- 84 training samples
- 21 test samples

Data is structured as:
```
data/
├── clinical_data/
│   ├── train_embeddings.npy
│   └── test_embeddings.npy
├── image_data/
│   ├── train_image_embeddings.npy
│   └── test_image_embeddings.npy
└── labels/
    ├── train_labels.csv
    └── test_labels.csv
```

## Example use cases

### Cancer Detection Pipeline

The main pipeline can be executed through the provided Jupyter notebooks:

1. **Standard pipeline** (pipeline.ipynb):
   - Processes multimodal patient data
   - Trains the GNN model
   - Evaluates performance on test set

2. **Attention-enhanced pipeline** (pipeline-att.ipynb):
   - Includes feature attention mechanism
   - Balances the importance of clinical vs. imaging data
   - Potentially improved performance for certain cancer types

### Performance Comparison

The model allows comparison between:
- Clinical-only classification
- Image-only classification
- Multimodal (combined) classification

This enables healthcare providers to understand the value added by integrating multiple data sources.

## Model Architecture

Our GNN architecture consists of:

- **Input**: Patient graphs with clinical and imaging nodes
- **Graph Convolution Layers**: Process node features while accounting for graph structure
- **Attention Mechanisms**: Weight feature importance (in att-version)
- **Pooling Layer**: Aggregate node representations into a patient-level representation
- **Classification Layer**: Binary output (cancer/no-cancer)

## Results

The model achieves strong performance metrics on the test set, with detailed evaluation available in the notebooks including:
- Accuracy
- Precision
- Recall
- F1-Score

## Future Work

- Integration with quantum computing techniques for enhanced feature processing
- Expansion to multi-class classification for cancer type identification
- Incorporation of additional modalities (genomic data, patient history)
- Explainability enhancements for clinical decision support

## Contributors

Team 01 - The Blaise Pascal Quantum Challenge

## License

MIT License