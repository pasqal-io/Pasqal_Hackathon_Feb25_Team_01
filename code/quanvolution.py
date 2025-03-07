import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import pydicom
from qadence import Backend, rydberg_hea

class DICOMDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        
        # No specific transformations here, just converting to tensor
        base_transform = [
            transforms.ToTensor(),
            transforms.Resize((36,36))
        ]
        self.transform = transforms.Compose(base_transform)

        # Iterate through the directories (patients) and find the DICOM file
        patient_dirs = sorted(os.listdir(root_dir))  # Assuming each patient has a folder
        for patient_dir in patient_dirs:
            patient_path = os.path.join(root_dir, patient_dir)
            if os.path.isdir(patient_path):
                dicom_file = self.find_dicom_file(patient_path)
                if dicom_file:
                    self.image_paths.append(dicom_file)
                    self.labels.append(int(patient_dir))  # Assuming patient directory is the label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        dicom_file = self.image_paths[idx]
        dicom_data = pydicom.dcmread(dicom_file)
        image = dicom_data.pixel_array  # Get pixel data as numpy array
        
        # Normalize and convert to tensor
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  
        return image, label

    def find_dicom_file(self, directory):
        for file in os.listdir(directory):
            if file.endswith(".dcm"):
                return os.path.join(directory, file)
        return None

def get_qadence_device(n_atoms):
    lattice_positions = np.array([ 
        [i, j] for i in range(int(np.sqrt(n_atoms))) for j in range(int(np.sqrt(n_atoms)))
    ]) 
    return lattice_positions

def define_qadence_circuit(n_atoms, pulse_params):
    lattice = get_qadence_device(n_atoms)
    
    def circuit(patch):
        h = Backend(lattice)
        
        for i in range(n_atoms):
            h.add_pulse(rydberg_hea.rabi, patch[i] * pulse_params[i], i)
            h.add_pulse(rydberg_hea.detuning, -patch[i] * pulse_params[i], i)
        
        h.evolve(time=0.5)
        results = h.measure()  
        
        return results
    
    return circuit

def qadence_quanvolution(image, circuit, patch_size, n_atoms):
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    
    height_patches = image.shape[0] // patch_size
    width_patches = image.shape[1] // patch_size
    out = np.zeros((height_patches, width_patches, n_atoms * 32)) 
    
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
            for c in range(n_atoms * 32):
                out[j, k, c] = q_results[c % n_atoms] 
    
    return out

def qadence_quanvolution_batch(images, circuit, patch_size, n_atoms):
    batch_size = images.shape[0]
    processed = [
        qadence_quanvolution(images[i].detach().cpu().numpy(), circuit, patch_size, n_atoms)
        for i in range(batch_size)
    ]
    
    processed = np.array(processed)
    return torch.tensor(processed, dtype=torch.float32).to(images.device)

n_atoms = 4 
pulse_params = np.random.uniform(0, 2 * np.pi, size=n_atoms)
qadence_circuit = define_qadence_circuit(n_atoms, pulse_params)