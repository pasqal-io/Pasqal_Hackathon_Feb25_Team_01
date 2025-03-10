import sys
import os
import pandas as pd
import numpy as np
import logging
import torch
from torch.utils.data import Dataset

# Dataset class for precomputed embeddings
class PrecomputedEmbeddingsDataset(Dataset):
    def __init__(self, img_embeddings_dir, clinical_embeddings_dir, labels_file, isTraining=True):
        self.img_embeddings_dir = img_embeddings_dir
        self.clinical_embeddings_dir = clinical_embeddings_dir
        self.labels_file = labels_file
        self.isTraining = isTraining

        # Load PRE-TRAINED embeddings
        # CLINICAL EMBEDDINGS
        if os.path.exists(self.clinical_embeddings_dir):
            self.clinical_embds = np.load(self.clinical_embeddings_dir)
            self.clinical_embds = torch.tensor(self.clinical_embds, dtype=torch.float32)
            logging.info("Loaded clinical embeddings of size %s", self.clinical_embds.size())
        else:
            raise FileNotFoundError(f"Clinical embeddings not found at {self.clinical_embeddings_dir}")
        
        # IMAGE EMBEDDINGS
        if os.path.exists(self.img_embeddings_dir):
            self.image_embds= np.load(self.img_embeddings_dir)
            self.image_embds = torch.tensor(self.image_embds, dtype=torch.float32)
            logging.info("Loaded clinical embeddings of size %s", self.image_embds.size())
        else:
            raise FileNotFoundError(f"Clinical embeddings not found at {self.img_embeddings_dir}")

        # Target labels
        if os.path.exists(self.img_embeddings_dir):
            self.labels_df = pd.read_csv(self.labels_file) 
            self.labels = self.labels_df.iloc[:, 1].tolist()
            self.patient_ids = self.labels_df.iloc[:, 0].tolist()
            logging.info(f"Total Number of Patients: {len(self.patient_ids)}")
        else:
            raise FileNotFoundError(f"Labels not found at {self.labels_file}")
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.patient_ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.patient_ids[index]
        logging.info(f"ID: {ID}")

        clinical_embds = self.clinical_embds[index, :, :]
        image_embds = self.image_embds[index, :, :]
        logging.info("Clinical embeddings of size %s", clinical_embds.size())
        logging.info("Image embeddings of size %s", image_embds.size())
            
        # Convert Labels to tensor
        labels_torch = torch.tensor(self.labels[index], dtype=torch.float32)

        if self.isTraining:
            return image_embds, clinical_embds, labels_torch
        else:
            return image_embds, clinical_embds, labels_torch, ID