import sys
import os
import pandas as pd
import numpy as np
import logging
import torch
from torch.utils.data import Dataset

# Dataset class for precomputed embeddings
class PrecomputedEmbeddingsDataset(Dataset):
    def __init__(self, img_embeddings_dir, clinical_embeddings_dir, labels_file, fold_splits, fold_id, isTraining=True):
        # Check valid data directories
        if not os.path.exists(img_embeddings_dir):
            sys.exit("Invalid image embeddings directory %s" % img_embeddings_dir)
        if not os.path.exists(clinical_embeddings_dir):
            sys.exit("Invalid clinical embeddings directory %s" % clinical_embeddings_dir)
        if not os.path.exists(labels_file):
            sys.exit("Invalid feature labels path %s" % labels_file)

        self.img_embeddings_dir = img_embeddings_dir
        self.clinical_embeddings_dir = clinical_embeddings_dir
        self.labels_file = labels_file
        self.isTraining = isTraining

        # Target labels
        self.labels_df = pd.read_csv(self.labels_file) 

        # Get patients ids from labels file
        self.patient_ids = self.labels_df.iloc[:, 0].tolist()
        logging.info(f"Total Number of Patients: {len(self.patient_ids)}")

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.patient_ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.patient_ids[index]
        logging.info(f"ID: {ID}")
        
        # Load PRE-TRAINED embeddings
        # CLINICAL EMBEDDINGS
        if os.path.exists(self.clinical_embeddings_dir):
            clinical_embds = np.load(self.clinical_embeddings_dir)
            clinical_embds = torch.tensor(clinical_embds, dtype=torch.float32)
            clinical_embds = clinical_embds[index, :, :]
            logging.info("Loaded clinical embeddings of size %s", clinical_embds.size())
        else:
            raise FileNotFoundError(f"Clinical embeddings not found at {self.clinical_embeddings_dir}")
        
        # IMAGE EMBEDDINGS
        if os.path.exists(self.img_embeddings_dir):
            image_embds= np.load(self.img_embeddings_dir)
            image_embds = torch.tensor(image_embds, dtype=torch.float32)
            image_embds = image_embds[index, :, :]
            logging.info("Loaded clinical embeddings of size %s", image_embds.size())
        else:
            raise FileNotFoundError(f"Clinical embeddings not found at {self.img_embeddings_dir}")
            
        # Labels (currenly only one label is used for binary classification)
        labels = self.labels_df.loc[index].tolist()

        # Convert Labels to tensor
        labels_torch = torch.tensor(labels[1], dtype=torch.float32)

        if self.isTraining:
            return image_embds, clinical_embds, labels_torch
        else:
            return image_embds, clinical_embds, labels_torch, ID