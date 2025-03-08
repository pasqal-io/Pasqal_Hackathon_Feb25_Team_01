import numpy as np
import logging
import sys
import torch
import os
from PIL import Image
import qadence as qd
from torch.utils.data import Dataset
import torchvision.transforms as transforms

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

class PNGDataset(Dataset):
    def __init__(self, root_dir):
        self.data = {}  # Using a dictionary to map patient_id to image and label

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((36, 36)), 
            transforms.ToTensor()
        ])

        # Get all files in the directory and sort them based on the patient ID extracted from the filename
        for file in sorted(os.listdir(root_dir), key=lambda x: int(os.path.splitext(x)[0])):
            if file.endswith(".png"):
                patient_id = int(os.path.splitext(file)[0])  # Extract patient ID as integer
                image_path = os.path.join(root_dir, file)
                # Store the image path and label in the dictionary
                self.data[patient_id] = {"image_path": image_path, "label": patient_id}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Access the data using the patient_id (idx) as key
        patient_data = self.data[idx]
        image = Image.open(patient_data["image_path"])
        image = self.transform(image)
        label = torch.tensor(patient_data["label"], dtype=torch.long)
        return image, label, patient_data["image_path"]

class QCNN:
    def __init__(self, n_atoms=4):
        self.n_atoms = n_atoms
        self.pulse_params = np.random.uniform(0, 2 * np.pi, size=n_atoms)
        self.qadence_circuit = self.define_qadence_circuit()
        logging.info("Generating Image Embeddings with Quantum Processing...")

    def define_qadence_circuit(self):
        # Quantum Register definition (adjusted for line and spacing)
        n_qubits = self.n_atoms
        register = qd.Register.line(n_qubits, spacing=8.0)

        # Feature map with analog rotations
        x = qd.FeatureParameter("x")
        fm = qd.chain(
            qd.AnalogRX(x),
            qd.AnalogRY(2 * x),
            qd.AnalogRZ(3 * x)
        )

        # Define variational parameters for ansatz
        t_0 = 1000. * qd.VariationalParameter("t_0")
        t_1 = 1000. * qd.VariationalParameter("t_1")
        t_2 = 1000. * qd.VariationalParameter("t_2")

        # Ansatz with parameterized rotations and analog interactions
        ansatz = qd.chain(
            qd.AnalogRX("tht_0"),
            qd.AnalogRY("tht_1"),
            qd.AnalogRZ("tht_2"),
            qd.AnalogInteraction(t_0),
            qd.AnalogRX("tht_3"),
            qd.AnalogRY("tht_4"),
            qd.AnalogRZ("tht_5"),
            qd.AnalogInteraction(t_1),
            qd.AnalogRX("tht_6"),
            qd.AnalogRY("tht_7"),
            qd.AnalogRZ("tht_8"),
            qd.AnalogInteraction(t_2)
        )

        # Total magnetization observable
        observable = qd.hamiltonian_factory(n_qubits, detuning=qd.Z)

        # Quantum Circuit definition
        circuit = qd.QuantumCircuit(register, fm, ansatz)

        return circuit

    def simulate_quantum_operation(self, patch):
        # Simula a operação quântica no circuito com os parâmetros passados
        patch_tensor = qd.Tensor(np.array(patch).reshape(-1, 1))  # Ajustar o patch para tensor
        results = self.qadence_circuit(patch_tensor)  # Executar o circuito
        return results.numpy()  # Retornar os resultados da operação

    def define_simplified_circuit(self):
        # Função para simplificar a definição do circuito
        def circuit(patch):
            return self.simulate_quantum_operation(patch)
        return circuit

    def qadence_quanvolution(self, image, circuit, patch_size):
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        height_patches = image.shape[0] // patch_size
        width_patches = image.shape[1] // patch_size
        out = np.zeros((height_patches, width_patches, 128))

        for j in range(height_patches):
            for k in range(width_patches):
                patch = []
                for i in range(patch_size):
                    for l in range(patch_size):
                        if (j * patch_size + i < image.shape[0]) and (k * patch_size + l < image.shape[1]):
                            patch.append(image[j * patch_size + i, k * patch_size + l, 0])
                        else:
                            patch.append(0)

                q_results = circuit(patch)
                for c in range(128):
                    out[j, k, c] = q_results[c % self.n_atoms]

        return out
    
    def process_and_save_embeddings(self, png_dataset, circuit, patient_ids, patch_size, isTrain=True):
        embeddings_list = []
        
        logging.info("Generating Image Embeddings...")
        for patient_id in patient_ids:
            image, label, _ = png_dataset[patient_id]
            image = image.squeeze(0).numpy()
            full_embedding = self.qadence_quanvolution(image, circuit, patch_size)
            
            embeddings_list.append(full_embedding)
            logging.info("Patient %s and label %s ", patient_id, label)
        
        embeddings_array = np.array(embeddings_list)
        
        # Save embeddings
        if isTrain: 
            logging.info("Train Embeddings shape: %s", embeddings_array.shape)
            np.save("data/image_data/train_embeddings.npy", embeddings_array)
        else:
            logging.info("Test Embeddings shape: %s", embeddings_array.shape)
            np.save("data/image_data/test_embeddings.npy", embeddings_array)

        logging.info("Image Embeddings saved successfully!")
