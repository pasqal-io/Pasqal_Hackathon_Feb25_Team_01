from quanvolution import QCNN, PNGDataset
import torch
import logging

# Load dataset
png_dataset = PNGDataset("data/image_data/png")
patient_ids = list(png_dataset.data.keys())

# Initialize QCNN
qcnn = QCNN(n_atoms=4)
circuit = qcnn.define_qadence_circuit()

# Training data
torch.manual_seed(42)
x_train = torch.linspace(-1.0, 1.0, steps=10)
y_train = x_train**2

# Train model once
logging.info("Training model...")
qcnn.train_model(x_train, y_train, epochs=40, lr=0.1)

# Generate embeddings for training set
train_ids = patient_ids[:len(patient_ids)//2]
test_ids = patient_ids[len(patient_ids)//2:]

qcnn.process_and_save_embeddings(png_dataset, circuit, train_ids, patch_size=6, filename="train_embeddings.npy")

qcnn.process_and_save_embeddings(png_dataset, circuit, test_ids, patch_size=6, filename="test_embeddings.npy")