import logging
import sys

import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss
import matplotlib.pyplot as plt

from src.rmrm.model.network import Network
from src.rmrm.embeddings import PrecomputedEmbeddingsDataset
from src.rmrm.utils.utils import *


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def plot_losses(self, train_losses, save_dir):
        # Plot Training Loss Curve
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b', label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.savefig(save_dir + 'plot_loss.png')
        plt.show()

    def train(self):
        json_opts = json_file_to_pyobj(self.args.config)

        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                            level=logging.INFO,
                            stream=sys.stdout)

        # Create experiment directories
        if self.args.resume_epoch == None:
            make_new = True 
        else:
            make_new = False
        timestamp = get_experiment_id(make_new, json_opts.experiment_dirs.load_dir)
        experiment_path = 'results' + '/' + timestamp
        make_dir(experiment_path + '/' + json_opts.experiment_dirs.model_dir)

        # Set up the model
        logging.info("Initialising model")
        n_out_features = 1

        # Initialize the model
        model = Network(n_out_features,
                        json_opts.training_params.batch_size, 
                        self.device) 
        model = model.to(self.device)

        # Dataloader for precomputed embeddings
        logging.info("Preparing data")
        num_workers = json_opts.data_params.num_workers
        
        # Use the dataset class with precomputed embeddings
        train_dataset = PrecomputedEmbeddingsDataset(
            json_opts.data_source.img_embeddings_dir,
            json_opts.data_source.clinical_embeddings_dir,
            json_opts.data_source.labels_file,
            isTraining=True
        )
        
        train_loader = DataLoader(dataset=train_dataset, 
                                batch_size=json_opts.training_params.batch_size, 
                                shuffle=False,
                                num_workers=num_workers, 
                                drop_last=True)

        n_train_examples = len(train_loader)
        logging.info("Total number of training examples: %d" % n_train_examples)

        # Loss and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=json_opts.training_params.learning_rate, 
                                    betas=(json_opts.training_params.beta1, 
                                            json_opts.training_params.beta2),
                                    weight_decay=json_opts.training_params.l2_reg_alpha)

        if self.args.resume_epoch != None:
            initial_epoch = self.args.resume_epoch
        else:
            initial_epoch = 0

        # Restore saved model
        if self.args.resume_epoch != None:
            load_path = experiment_path + '/' + json_opts.experiment_dirs.model_dir + \
                        "/epoch_%d.pth" % (self.args.resume_epoch)
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            assert(epoch == self.args.resume_epoch)
            logging.info("Resume training, successfully loaded %s ", load_path)

        logging.info("Begin training")

        model = model.train()

        train_losses = []

        for epoch in range(initial_epoch, json_opts.training_params.total_epochs):
            epoch_train_loss = 0.

            for _, (img_embeddings, clinical_embeddings, batch_y) in enumerate(train_loader):

                # Transfer to GPU
                img_embeddings, clinical_embeddings = img_embeddings.to(self.device), clinical_embeddings.to(self.device)
                batch_y = batch_y.to(self.device)
                
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
            train_losses.append(avg_train_loss)
            logging.info("Epoch %d - Average Binary Classification Loss: %.4f" % (epoch+1, avg_train_loss))

            # Print training loss every epoch
            logging.info('Epoch[{}/{}], total loss:{:.4f}'.format(epoch+1, json_opts.training_params.total_epochs, epoch_train_loss))

        # Save the model only at the last epoch
        final_save_path = experiment_path + '/' + json_opts.experiment_dirs.model_dir + "/final_model.pth"
        torch.save({'epoch': json_opts.training_params.total_epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, final_save_path)
        logging.info("Final model saved: %s" % final_save_path)

        logging.info("Training finished")

        # plot loss curve
        plot_save_dir = experiment_path + '/' + json_opts.experiment_dirs.model_dir + '/plot'
        self.plot_losses(train_losses, plot_save_dir)
