import numpy as np
import torch
import logging
import sys
import os
import qadence as qd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

class PNGDataset(Dataset):
    def __init__(self, root_dir):
        self.data = {}
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((36, 36)), 
            transforms.ToTensor()
        ])
        for file in sorted(os.listdir(root_dir), key=lambda x: int(os.path.splitext(x)[0])):
            if file.endswith(".png"):
                patient_id = int(os.path.splitext(file)[0])
                image_path = os.path.join(root_dir, file)
                self.data[patient_id] = {"image_path": image_path, "label": patient_id}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient_data = self.data[idx]
        image = Image.open(patient_data["image_path"])
        image = self.transform(image)
        label = torch.tensor(patient_data["label"], dtype=torch.long)
        return image, label, patient_data["image_path"]

class QCNN:
    def __init__(self, n_atoms=4):
        self.n_atoms = n_atoms
        self.qadence_circuit = self.define_qadence_circuit()
        if self.qadence_circuit is None:
            raise ValueError("Quantum circuit initialization failed!")
        logging.info("Quantum model initialized successfully!")

    def define_qadence_circuit(self):
        n_qubits = self.n_atoms
        register = qd.Register.line(n_qubits, spacing=8.0)

        x = qd.FeatureParameter("x")
        fm = qd.chain(
            qd.AnalogRX(x),
            qd.AnalogRY(2 * x),
            qd.AnalogRZ(3 * x)
        )

        ansatz = qd.chain(
            qd.AnalogRX("tht_0"),
            qd.AnalogRY("tht_1"),
            qd.AnalogRZ("tht_2"),
            qd.AnalogInteraction("t_0"),
            qd.AnalogRX("tht_3"),
            qd.AnalogRY("tht_4"),
            qd.AnalogRZ("tht_5"),
            qd.AnalogInteraction("t_1")
        )

        observable = qd.hamiltonian_factory(n_qubits, detuning=qd.Z)
        circuit = qd.QuantumCircuit(register, fm, ansatz)

        model = qd.QuantumModel(
            circuit,
            observable=observable,
            backend=qd.BackendName.PYQTORCH,
            diff_mode=qd.DiffMode.AD
        )
        return model


    def qadence_quanvolution(self, image, circuit, patch_size, n_atoms):
        height_patches = image.shape[0] // patch_size
        width_patches = image.shape[1] // patch_size
        out = np.zeros((height_patches, width_patches, 128))  
        
        for j in range(height_patches):
            for k in range(width_patches):
                patch = []
                for i in range(patch_size):
                    for l in range(patch_size):
                        if (j * patch_size + i < image.shape[0]) and (k * patch_size + l < image.shape[1]):
                            patch.append(image[j * patch_size + i, k * patch_size + l])
                        else:
                            patch.append(0)

                patch_tensor = torch.tensor(patch, dtype=torch.float32).reshape(1, patch_size, patch_size)
                
                patch_tensor = patch_tensor.unsqueeze(0) 
                
                q_results = circuit({"x": patch_tensor})

                for c in range(128):
                    result = q_results[c % n_atoms]
                    if result.numel() > 1:
                        # Handle complex numbers by taking the real part or magnitude
                        result = result.mean()
                        if result.is_complex():
                            result = result.real
                        out[j, k, c] = result.item()
                    else:
                        # Handle scalar case (real or complex)
                        if result.is_complex():
                            out[j, k, c] = result.real.item()
                        else:
                            out[j, k, c] = result.item()
        
        return out

    def train_model(self, x_train, y_train, epochs=20, lr=0.1):
        mse_loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.qadence_circuit.parameters(), lr=lr)
        
        def loss_fn(x, y):
            out = self.qadence_circuit.expectation({"x": x})
            return mse_loss(out.squeeze(), y)

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(x_train, y_train)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                logging.info("Epoch %d, Loss: %.4f", epoch, loss.item())

    def process_and_save_embeddings(self, png_dataset, circuit, patient_ids, patch_size, filename):
        embeddings_list = []
        logging.info("Generating Image Embeddings...")
        
        for patient_id in patient_ids:
            image, label, _ = png_dataset[patient_id]
            image = image.squeeze(0).numpy()
            full_embedding = self.qadence_quanvolution(image, circuit, patch_size, self.n_atoms)
            embeddings_list.append(full_embedding)
            logging.info("Patient %s and label %s", patient_id, label)
        
        embeddings_array = np.array(embeddings_list)
        np.save(filename, embeddings_array)
        logging.info("Embeddings saved: %s", filename)