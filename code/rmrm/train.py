import argparse
import logging
import sys
import os
import pickle
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCELoss  # Import BCELoss for binary classification

from model.network import Network
from utils.utils import *

import numpy as np

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
    if config.resume_epoch == None:
        make_new = True 
    else:
        make_new = False
    timestamp = get_experiment_id(make_new, json_opts.experiment_dirs.load_dir, config.fold_id)
    experiment_path = 'experiments' + '/' + timestamp
    make_dir(experiment_path + '/' + json_opts.experiment_dirs.model_dir)

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
    train_dataset = PrecomputedEmbeddingsDataset(
        json_opts.data_source.img_embeddings_dir,
        json_opts.data_source.clinical_embeddings_dir,
        json_opts.data_source.labels_file,
        json_opts.data_source.fold_splits,
        config.fold_id,
        isTraining=True
    )
    
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=json_opts.training_params.batch_size, 
                              shuffle=True, num_workers=num_workers, drop_last=True)

    n_train_examples = len(train_loader)
    logging.info("Total number of training examples: %d" % n_train_examples)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=json_opts.training_params.learning_rate, 
                                 betas=(json_opts.training_params.beta1, 
                                        json_opts.training_params.beta2),
                                 weight_decay=json_opts.training_params.l2_reg_alpha)

    if config.resume_epoch != None:
        initial_epoch = config.resume_epoch
    else:
        initial_epoch = 0

    # Restore saved model
    if config.resume_epoch != None:
        load_path = experiment_path + '/' + json_opts.experiment_dirs.model_dir + \
                    "/epoch_%d.pth" % (config.resume_epoch)
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        assert(epoch == config.resume_epoch)
        print("Resume training, successfully loaded " + load_path)

    logging.info("Begin training")

    model = model.train()

    for epoch in range(initial_epoch, json_opts.training_params.total_epochs):
        epoch_train_loss = 0.

        for _, (img_embeddings, clinical_embeddings, batch_y, death_indicator) in enumerate(train_loader):

            # Transfer to GPU
            img_embeddings, clinical_embeddings = img_embeddings.to(device), clinical_embeddings.to(device)
            batch_y = batch_y.to(device)
            death_indicator = death_indicator.to(device)
            
            # Reshape embeddings for more efficient processing
            # Ensure image embeddings match expected dimensions
            img_embeddings = img_embeddings.view(json_opts.training_params.batch_size, model.n_pixel, -1)
            
            # Ensure clinical embeddings match expected dimensions
            clinical_embeddings = clinical_embeddings.view(json_opts.training_params.batch_size, model.n_clinical, -1)
            
            optimizer.zero_grad()

            # Forward pass with precomputed embeddings
            final_pred = model.forward(clinical_embeddings, img_embeddings)

            # Optimisation
            if torch.sum(death_indicator) > 0.0:
                # Binary cross-entropy loss for binary classification
                loss = BCELoss()(final_pred, death_indicator.float())
                
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.detach().cpu().numpy()
           
                
        # Log training loss
        avg_train_loss = epoch_train_loss / len(train_loader)
        logging.info("Epoch %d - Average Binary Classification Loss: %.4f" % (epoch+1, avg_train_loss))
                
        # Save model
        if (epoch % json_opts.save_freqs.model_freq) == 0:
            save_path = experiment_path + '/' + json_opts.experiment_dirs.model_dir + \
                        "/epoch_%d.pth" % (epoch+1)
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, save_path)
            logging.info("Model saved: %s" % save_path)

        # Print training loss every epoch
        print('Epoch[{}/{}], total loss:{:.4f}'.format(epoch+1, json_opts.training_params.total_epochs, 
                                                       epoch_train_loss))

    logging.info("Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', default='configs/config', type=str,
                        help='config file path')
    parser.add_argument('--resume_epoch', default=None, type=int,
                        help='resume training from this epoch, set to None for new training')
    parser.add_argument('--fold_id', default=1, type=int,
                        help='which cross-validation fold')

    config = parser.parse_args()
    main(config)
