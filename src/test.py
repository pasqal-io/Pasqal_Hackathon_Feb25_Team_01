import logging
import sys
import pickle

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Import classification metrics

from src.rmrm.model.network import Network
from src.rmrm.embeddings import PrecomputedEmbeddingsDataset
from src.rmrm.utils.utils import *

import numpy as np


class Evaluator: 
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test(self):
        json_opts = json_file_to_pyobj(self.args.config)

        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                            level=logging.INFO,
                            stream=sys.stdout)

        # Create experiment directories
        make_new = False
        timestamp = get_experiment_id(make_new, json_opts.experiment_dirs.load_dir, self.args.fold_id)
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
        model = Network(model_opts, 
                        n_out_features, 
                        json_opts.training_params.batch_size, 
                        self.device)
        model = model.to(self.device)

        # Dataloader for precomputed embeddings
        logging.info("Preparing data")
        num_workers = json_opts.data_params.num_workers
        
        # Use the dataset class with precomputed embeddings
        test_dataset = PrecomputedEmbeddingsDataset(
            json_opts.data_source.img_embeddings_dir,
            json_opts.data_source.clinical_embeddings_dir,
            json_opts.data_source.labels_file,
            json_opts.data_source.fold_splits,
            self.args.fold_id,
            isTraining=False
        )
        
        test_loader = DataLoader(dataset=test_dataset, 
                                batch_size=json_opts.training_params.batch_size, 
                                shuffle=True, 
                                num_workers=num_workers, 
                                drop_last=True)

        # Load model
        load_path = model_dir + "/epoch_%d.pth" % (self.args.test_epoch)
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded " + load_path)

        model = model.eval()

        # Evaluation
        all_preds = []
        all_labels = []
        all_ids = []

        with torch.no_grad():
            for _, (img_embeddings, clinical_embeddings, batch_y, id) in enumerate(test_loader):
                # Transfer to GPU
                img_embeddings, clinical_embeddings = img_embeddings.to(self.device), clinical_embeddings.to(self.device)
                batch_y = batch_y.to(self.device)

                # Reshape embeddings for more efficient processing
                # Ensure image embeddings match expected dimensions
                img_embeddings = img_embeddings.view(json_opts.training_params.batch_size, model.n_pixel, -1)
                
                # Ensure clinical embeddings match expected dimensions
                clinical_embeddings = clinical_embeddings.view(json_opts.training_params.batch_size, model.n_clinical, -1)

                # Forward pass with precomputed embeddings
                final_pred = model.forward(clinical_embeddings, img_embeddings)

                # Store predictions and labels
                all_preds.append(final_pred.detach().cpu().numpy())
                all_labels.append(batch_y.detach().cpu().numpy())
                all_ids.append(id)

        # Concatenate results
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Calculate C-index
        c_index = concordance_index(all_labels, all_preds)
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
            'c_index': c_index,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        save_path = test_output_dir + "/results_epoch_%d.pkl" % (self.args.test_epoch)
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        logging.info("Results saved to %s" % save_path)
