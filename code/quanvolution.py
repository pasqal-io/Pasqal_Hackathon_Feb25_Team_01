import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
from qadence import Backend, rydberg_hea

class PNGDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((36, 36)),  # Tamanho 36x36
            transforms.ToTensor()
        ])

        for file in sorted(os.listdir(root_dir)):
            if file.endswith(".png"):
                self.image_paths.append(os.path.join(root_dir, file))
                patient_id = int(os.path.splitext(file)[0])  # Patient ID from filename
                self.labels.append(patient_id)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label, self.image_paths[idx]

def simulate_quantum_operation(patch, pulse_params, n_atoms):
    # Simplified quantum simulation (classical approximation)
    results = np.zeros(n_atoms)
    for i in range(n_atoms):
        # Simulate some transformation based on the pulse parameters
        results[i] = np.sum(patch) * pulse_params[i]  # Just an example computation
    return results

def define_simplified_circuit(n_atoms, pulse_params):
    # Instead of using a quantum backend, we directly simulate the circuit's behavior
    def circuit(patch):
        return simulate_quantum_operation(patch, pulse_params, n_atoms)
    return circuit

def qadence_quanvolution(image, circuit, patch_size, n_atoms):
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    
    height_patches = image.shape[0] // patch_size
    width_patches = image.shape[1] // patch_size
    out = np.zeros((height_patches, width_patches, 128))  # Placeholder size for embedding
    
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
                out[j, k, c] = q_results[c % n_atoms] 
    
    return out

def save_embeddings(embeddings, patient_ids, filename):
    # Flatten the embeddings and create a DataFrame
    flat_embeddings = embeddings.reshape(-1, 128)  # Flatten to rows x 128
    
    # Create the header with 'Patient' followed by columns 0, 1, ..., 127
    header = ['Patient'] + [str(i) for i in range(128)]
    
    # Initialize a list to store patient IDs for each embedding row
    all_patient_ids = []

    # Determine the number of patches per image and repeat patient_id accordingly
    for i, patient_id in enumerate(patient_ids):
        num_patches = flat_embeddings.shape[0] // len(patient_ids)  # Get number of embeddings per patient
        all_patient_ids.extend([patient_id] * num_patches)

    # Ensure that the length of patient_ids matches the number of embedding rows
    assert len(all_patient_ids) == flat_embeddings.shape[0], "Patient ID list does not match embedding rows"

    # Save as CSV with the correct header
    df = pd.DataFrame(flat_embeddings)
    df.insert(0, 'Patient', all_patient_ids)  # Add patient column with corresponding IDs
    df.to_csv(filename, index=False, header=header)

def process_and_save_embeddings(dataset, circuit, patch_size, n_atoms, output_file):
    embeddings_list = []  
    patient_ids = []  # Store patient IDs to be used in the CSV file
    
    for i in range(len(dataset)):
        image, label, _ = dataset[i]
        image = image.squeeze(0).numpy()
        embeddings = qadence_quanvolution(image, circuit, patch_size, n_atoms)
        
        embeddings_flattened = embeddings.flatten()  # Flatten the embeddings
        embeddings_list.append(embeddings_flattened)
        
        # Append the patient ID for the current image (for each embedding row)
        patient_ids.append(label.item())
    
    # Convert the embeddings list to a numpy array for easier manipulation
    embeddings_array = np.array(embeddings_list)
    
    # Save the embeddings with the correct patient IDs
    save_embeddings(embeddings_array, patient_ids, output_file)

# Example usage
n_atoms = 4 
pulse_params = np.random.uniform(0, 2 * np.pi, size=n_atoms)
simplified_circuit = define_simplified_circuit(n_atoms, pulse_params)

dataset = PNGDataset("/home/eflammere/Pasqal_Hackathon_Feb25_Team_01/data/png")
process_and_save_embeddings(dataset, simplified_circuit, patch_size=6, n_atoms=n_atoms, output_file="embeddings.csv")
