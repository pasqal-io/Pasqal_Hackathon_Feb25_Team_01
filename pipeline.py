from sklearn.model_selection import train_test_split
from data.clinical_data.clinical_data_embeddings import DataPreprocessing, ClinicalDataEmbeddings
from src.train import Trainer
from src.test import Evaluator
import pandas as pd
import argparse
import logging


class Pipeline:
    def __init__(self, args):
        self.args = args
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def generate_data(self):
        # Defining the target variable [Censored_0_progressed_1 → 1: Indicates liver cancer and 0: Indicates no progression or absence of liver cancer]
        target_column = "Censored_0_progressed_1"
        id_column = "TCIA_ID"
        embedding_dim = 128

        # Load and preprocess data
        data_preprocessing = DataPreprocessing("data/clinical_data/clinical_Data.csv")
        patient_data, patient_labels, patient_ids = data_preprocessing.preprocesses_data()

        # Split into train & test sets (80/20 split)
        train_data, test_data, train_labels, test_labels, train_ids, test_ids = train_test_split(
            patient_data, patient_labels, patient_ids, test_size=0.2, random_state=42, stratify=patient_labels
        )

        # Save train/test sets and corresponding labels as CSV 
        train_data_df = pd.DataFrame(train_data.numpy())
        train_labels_df = pd.DataFrame({id_column: train_ids,  target_column: train_labels.numpy()})
        train_data_df.to_csv("data/train_data.csv", index=False)
        train_labels_df.to_csv("data/labels/train_labels.csv", index=False)

        test_data_df = pd.DataFrame(test_data.numpy())
        test_labels_df = pd.DataFrame({id_column: test_ids, target_column: test_labels.numpy()})
        test_data_df.to_csv("data/test_data.csv", index=False)
        test_labels_df.to_csv("data/labels/test_labels.csv", index=False)
        
        # Generate Clinical Embeddings
        embedding_generator = ClinicalDataEmbeddings(embedding_dim=embedding_dim, target_column=target_column)
        embedding_generator.train_model(train_data, train_labels)
        embedding_generator.generate_and_save_embeddings(train_data, train_labels, isTrain=True)
        embedding_generator.generate_and_save_embeddings(test_data, test_labels, isTrain=False)

        # TODO: Generate Image Embeddings

    def train(self):
        logging.info("Starting Training...")
        trainer = Trainer(self.args)
        trainer.train()

    def test(self):
        logging.info("Starting Testing...")
        evaluator = Evaluator(self.args)
        evaluator.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='src/config/config_train.json', help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["data", "train", "test"], required=True, help="Mode: data, train, or test")
    parser.add_argument('--resume_epoch', default=None, type=int, help='resume training from this epoch, set to None for new training')
    parser.add_argument('--test_epoch', default=50, type=int, help='test model from this epoch, -1 for last, -2 for all')
    parser.add_argument('--fold_id', default=1, type=int, help='which cross-validation fold')

    args = parser.parse_args()

    pipeline = Pipeline(args)
    if args.mode == "data":
        pipeline.generate_data()
    elif args.mode == "train":
        pipeline.train()
    elif args.mode == "test":
        pipeline.test()