from quanvolution import QCNN, PNGDataset
import torch
import logging


def test_model(qcnn, png_dataset, circuit, test_ids, patch_size, embeddings_filename):
    """
    Test the trained model using test embeddings.
    It re-generates embeddings from the test set and compares them with the saved test embeddings.
    """
    import numpy as np
    logging.info("Testing model on test embeddings...")

    computed_embeddings = []
    for patient_id in test_ids:
        image, label, _ = png_dataset[patient_id]
        # Remove channel dimension (if needed) and convert to numpy array
        image = image.squeeze(0).numpy()
        embedding = qcnn.qadence_quanvolution(image, circuit, patch_size, qcnn.n_atoms)
        computed_embeddings.append(embedding)
        logging.info("Processed test patient %s, label %s", patient_id, label)
    
    computed_embeddings = np.array(computed_embeddings)
    
    # Load saved test embeddings and compare with newly computed embeddings
    saved_embeddings = np.load(embeddings_filename)
    diff = np.mean(np.abs(computed_embeddings - saved_embeddings))
    logging.info("Mean absolute difference between computed and saved test embeddings: %.6f", diff)
    
    # If desired, assert a threshold for passing test (adjust threshold as needed)
    threshold = 1e-5
    if diff < threshold:
        logging.info("Test passed: Embeddings match within acceptable threshold.")
    else:
        logging.error("Test failed: Difference (%.6f) exceeds threshold (%.6f).", diff, threshold)

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

test_model(qcnn, png_dataset, circuit, test_ids, patch_size=6, embeddings_filename="test_embeddings.npy")