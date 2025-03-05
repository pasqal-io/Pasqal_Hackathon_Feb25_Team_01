import argparse
import logging
import os
import sys
import pickle
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Import classification metrics

from model.model import Network
from utils.utils import *

import numpy as np

# Reuse the PrecomputedEmbeddingsDataset from train.py
class PrecomputedEmbeddingsDataset(torch.utils.data.Dataset):
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

        # Data files
        if isTraining:
            patient_subset_txt = fold_splits + '/' + str(fold_id) + '_train.txt'
        else:
            patient_subset_txt = fold_splits + '/' + str(fold_id) + '_test.txt'

        # Get patient IDs from the split file
        with open(patient_subset_txt, 'r') as f:
            self.patient_ids = [line.strip() for line in f.readlines()]
        
        # Format patient IDs to match embedding filenames
        self.patient_ids = [os.path.basename(x).split('_')[0] for x in self.patient_ids]
        self.patient_ids = [x[:2] + '-' + x[2:] for x in self.patient_ids]

        # All the labels for this data subset
        data_df = pd.read_csv(self.labels_file)
        self.labels_df = data_df.iloc[:,:3]  # First 3 columns contain ID and survival data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.patient_ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.patient_ids[index]
        
        # Load pre-computed embeddings
        img_embedding_path = os.path.join(self.img_embeddings_dir, ID.replace('-', '') + '_img_embedding.pkl')
        clinical_embedding_path = os.path.join(self.clinical_embeddings_dir, ID.replace('-', '') + '_clinical_embedding.pkl')
        
        # Load embeddings using pickle
        with open(img_embedding_path, 'rb') as f:
            img_embedding = pickle.load(f)
        
        with open(clinical_embedding_path, 'rb') as f:
            clinical_embedding = pickle.load(f)
            
        # Get labels
        labels = self.labels_df.loc[self.labels_df['METABRIC.ID'] == ID].values.tolist()

        labels_time = np.zeros(1)
        labels_censored = np.zeros(1)

        labels_time[0] = labels[0][-1]
        labels_censored[0] = int(labels[0][-2])

        # Convert to tensor
        img_embedding_torch = torch.from_numpy(img_embedding).float()
        clinical_embedding_torch = torch.from_numpy(clinical_embedding).float()
        labels_torch = torch.from_numpy(labels_time).float()
        censored_torch = torch.from_numpy(labels_censored).long()

        if self.isTraining:
            return img_embedding_torch, clinical_embedding_torch, labels_torch, censored_torch
        else:
            return img_embedding_torch, clinical_embedding_torch, labels_torch, censored_torch, ID

def main(config):

    config_fold = config.config_file + str(config.fold_id) + '.json'
    json_opts = json_file_to_pyobj(config_fold)

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create experiment directories
    make_new = False
    timestamp = get_experiment_id(make_new, json_opts.experiment_dirs.load_dir, config.fold_id)
    experiment_path = 'experiments' + '/' + timestamp
    model_dir = experiment_path + '/' + json_opts.experiment_dirs.model_dir
    test_output_dir = experiment_path + '/' + json_opts.experiment_dirs.test_output_dir
    make_dir(test_output_dir)

    # Set up the model
    logging.info("Initialising model")
    model_opts = json_opts.model_params
    n_out_features = 1

    # Initialize the model - we'll still use the original Network class
    # but we'll skip the feature extraction parts in the forward pass
    model = Network(model_opts, n_out_features, 0,  # n_markers is not used when using precomputed embeddings
                    json_opts.training_params.batch_size, device,
                    0, [])  # n_cont_cols and n_classes_cat are not used when using precomputed embeddings
    model = model.to(device)

    # Dataloader for precomputed embeddings
    logging.info("Preparing data")
    num_workers = json_opts.data_params.num_workers
    
    # Use the dataset class with precomputed embeddings
    test_dataset = PrecomputedEmbeddingsDataset(
        json_opts.data_source.img_embeddings_dir,
        json_opts.data_source.clinical_embeddings_dir,
        json_opts.data_source.labels_file,
        json_opts.data_source.fold_splits,
        config.fold_id,
        isTraining=False
    )
    
    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=json_opts.training_params.batch_size, 
                             shuffle=False, num_workers=num_workers, drop_last=False)

    # Load model
    load_path = model_dir + "/epoch_%d.pth" % (config.epoch)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Successfully loaded " + load_path)

    model = model.eval()

    # Evaluation
    all_preds = []
    all_labels = []
    all_censored = []
    all_ids = []

    with torch.no_grad():
        for _, (img_embeddings, clinical_embeddings, batch_y, death_indicator, ids) in enumerate(test_loader):
            # Transfer to GPU
            img_embeddings, clinical_embeddings = img_embeddings.to(device), clinical_embeddings.to(device)
            batch_y = batch_y.to(device)
            death_indicator = death_indicator.to(device)

            # Get batch size (might be smaller for last batch)
            batch_size = img_embeddings.size(0)
            
            # Reshape embeddings for more efficient processing
            # Ensure image embeddings match expected dimensions
            img_embeddings = img_embeddings.view(batch_size, model.n_pixel, -1)
            
            # Ensure clinical embeddings match expected dimensions
            clinical_embeddings = clinical_embeddings.view(batch_size, model.n_clinical, -1)

            # Forward pass with precomputed embeddings
            final_pred = model.forward(clinical_embeddings, img_embeddings)

            # Store predictions and labels
            all_preds.append(final_pred.detach().cpu().numpy())
            all_labels.append(batch_y.detach().cpu().numpy())
            all_censored.append(death_indicator.detach().cpu().numpy())
            all_ids.extend(ids)

    # Concatenate results
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_censored = np.concatenate(all_censored, axis=0)

    # Calculate C-index
    c_index = concordance_index(all_labels, all_preds, all_censored)
    logging.info("C-index: %.4f" % c_index)

    # Calculate accuracy, precision, recall, and F1 score for binary classification
    accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    precision = precision_score(all_labels, (all_preds > 0.5).astype(int))
    recall = recall_score(all_labels, (all_preds > 0.5).astype(int))
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
    
    logging.info("Binary Classification Metrics:")
    logging.info("Accuracy: %.4f" % accuracy)
    logging.info("Precision: %.4f" % precision)
    logging.info("Recall: %.4f" % recall)
    logging.info("F1 Score: %.4f" % f1)

    # Save results
    results = {
        'predictions': all_preds,
        'labels': all_labels,
        'censored': all_censored,
        'ids': all_ids,
        'c_index': c_index,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    save_path = test_output_dir + "/results_epoch_%d.pkl" % (config.epoch)
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    logging.info("Results saved to %s" % save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', default='configs/config', type=str,
                        help='config file path')
    parser.add_argument('--epoch', default=100, type=int,
                        help='which epoch to test')
    parser.add_argument('--fold_id', default=1, type=int,
                        help='which cross-validation fold')

    config = parser.parse_args()
    main(config)
