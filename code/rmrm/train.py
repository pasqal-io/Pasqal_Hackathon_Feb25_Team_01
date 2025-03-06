import argparse
import logging
import sys
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCELoss                        # Import BCELoss for binary classification

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
        clinical_embedding_file = "data/Clinical data embeddings/top_9_clinical_embeddings.npy"
        if os.path.exists(clinical_embedding_file):
            clinical_embds = np.load(clinical_embedding_file)
            clinical_embds = torch.tensor(clinical_embds, dtype=torch.float32)
            clinical_embds = clinical_embds[index, :, :]
            logging.info("Loaded clinical embeddings of size %s", clinical_embds.size())
        else:
            raise FileNotFoundError(f"Clinical embeddings not found at {clinical_embedding_file}")
        
        # IMAGE EMBEDDINGS
        image_embedding_file = "data/Image data embeddings/image_embeddings_9_patients.npy"
        if os.path.exists(image_embedding_file):
            image_embds= np.load(image_embedding_file)
            image_embds = torch.tensor(image_embds, dtype=torch.float32)
            image_embds = image_embds[index, :, :]
            logging.info("Loaded clinical embeddings of size %s", image_embds.size())
        else:
            raise FileNotFoundError(f"Clinical embeddings not found at {image_embedding_file}")
            
        # Labels (currenly only one label is used for binary classification)
        labels = self.labels_df.loc[index].tolist()

        # Convert Labels to tensor
        labels_torch = torch.tensor(labels[1], dtype=torch.float32)

        if self.isTraining:
            return image_embds, clinical_embds, labels_torch
        else:
            return image_embds, clinical_embds, labels_torch, ID

def main(config):
    json_opts = json_file_to_pyobj(config.config_file)

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
    model = Network(model_opts, 
                    n_out_features,
                    json_opts.training_params.batch_size, 
                    device) 
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
                              shuffle=True, 
                              num_workers=num_workers, 
                              drop_last=True)

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

        for _, (img_embeddings, clinical_embeddings, batch_y) in enumerate(train_loader):

            # Transfer to GPU
            img_embeddings, clinical_embeddings = img_embeddings.to(device), clinical_embeddings.to(device)
            batch_y = batch_y.to(device)
            
            # Reshape embeddings for more efficient processing
            # Ensure image embeddings match expected dimensions
            img_embeddings = img_embeddings.view(json_opts.training_params.batch_size, model.n_pixel, -1)
            
            # Ensure clinical embeddings match expected dimensions
            clinical_embeddings = clinical_embeddings.view(json_opts.training_params.batch_size, model.n_clinical, -1)

            optimizer.zero_grad()

            # Forward pass with precomputed embeddings
            final_pred = model.forward(clinical_embeddings, img_embeddings)
            final_pred = final_pred.view(-1).float()        # reshape to match y dimentions and type

            # Binary cross-entropy loss for binary classification
            loss = BCELoss()(final_pred, batch_y)

            # Optimisation
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

    parser.add_argument('--config_file', default='code/rmrm/config/config.json', type=str,
                        help='config file path')
    parser.add_argument('--resume_epoch', default=None, type=int,
                        help='resume training from this epoch, set to None for new training')
    parser.add_argument('--fold_id', default=1, type=int,
                        help='which cross-validation fold')

    config = parser.parse_args()
    main(config)
