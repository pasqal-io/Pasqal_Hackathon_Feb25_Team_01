import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PNGDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((36, 36)), 
            transforms.ToTensor()
        ])

        for file in sorted(os.listdir(root_dir)):
            if file.endswith(".png"):
                self.image_paths.append(os.path.join(root_dir, file))
                patient_id = int(os.path.splitext(file)[0]) 
                self.labels.append(patient_id)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label, self.image_paths[idx]

def simulate_quantum_operation(patch, pulse_params, n_atoms):
    results = np.zeros(n_atoms)
    for i in range(n_atoms):
        results[i] = np.sum(patch) * pulse_params[i]
    return results

def define_simplified_circuit(n_atoms, pulse_params):
    def circuit(patch):
        return simulate_quantum_operation(patch, pulse_params, n_atoms)
    return circuit

def qadence_quanvolution(image, circuit, patch_size, n_atoms):
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
                out[j, k, c] = q_results[c % n_atoms]
    
    return out

def save_embeddings_npy(embeddings_array, patient_ids, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, patient_id in enumerate(patient_ids):
        patient_embedding = embeddings_array[i]
        np.save(os.path.join(output_dir, f"patient_{patient_id}.npy"), patient_embedding)

def process_and_save_embeddings(dataset, circuit, patch_size, n_atoms, output_dir):
    embeddings_list = []  
    patient_ids = [] 
    
    for i in range(len(dataset)):
        image, label, _ = dataset[i]
        image = image.squeeze(0).numpy()
        full_embedding = qadence_quanvolution(image, circuit, patch_size, n_atoms)
        
        embeddings_list.append(full_embedding)
        patient_ids.append(label.item())
    
    embeddings_array = np.array(embeddings_list)
    
    save_embeddings_npy(embeddings_array, patient_ids, output_dir)

n_atoms = 4 
pulse_params = np.random.uniform(0, 2 * np.pi, size=n_atoms)
simplified_circuit = define_simplified_circuit(n_atoms, pulse_params)

dataset = PNGDataset("/home/eflammere/Pasqal_Hackathon_Feb25_Team_01/data/png")
process_and_save_embeddings(dataset, simplified_circuit, patch_size=6, n_atoms=n_atoms, output_dir="embeddings_npy")