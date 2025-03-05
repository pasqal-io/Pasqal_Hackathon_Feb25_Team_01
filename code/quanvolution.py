import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from qadence import AnalogHamiltonianSimulation, rydberg

class LiverDataset(Dataset):
    def __init__(self, root_dir, train=True, data_augmentation=False):
        self.image_paths = []
        self.labels = []
        
        base_transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5186, 0.5187, 0.5185], std=[0.1957, 0.1957, 0.1957])
        ]

        augmentation_transform = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15)
        ]

        if train and data_augmentation:
            self.transform = transforms.Compose(augmentation_transform + base_transform)
        else:
            self.transform = transforms.Compose(base_transform)

        for label in ['0', '1']:
            folder_path = os.path.join(root_dir, label)
            images = sorted(os.listdir(folder_path)) 
            self.image_paths.extend([os.path.join(folder_path, img_name) for img_name in images])
            self.labels.extend([int(label)] * len(images))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  
        return image, label

def get_qadence_device(n_atoms):
    lattice_positions = np.array([
        [i, j] for i in range(int(np.sqrt(n_atoms))) for j in range(int(np.sqrt(n_atoms)))
    ]) 
    return lattice_positions

def define_qadence_circuit(n_atoms, pulse_params):
    lattice = get_qadence_device(n_atoms)
    
    def circuit(patch):
        h = AnalogHamiltonianSimulation(lattice)
        
        for i in range(n_atoms):
            h.add_pulse(rydberg.rabi, patch[i] * pulse_params[i], i)
            h.add_pulse(rydberg.detuning, -patch[i] * pulse_params[i], i)
        
        h.evolve(time=1.0)
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

root_dir = '/path/to/dataset' 
dataset = LiverDataset(root_dir=root_dir, train=True, data_augmentation=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

n_atoms = 4 
pulse_params = np.random.uniform(0, 2 * np.pi, size=n_atoms)
qadence_circuit = define_qadence_circuit(n_atoms, pulse_params)

for images, labels in dataloader:
    images = images.permute(0, 2, 3, 1).cpu().numpy() 
    output_embeddings = qadence_quanvolution_batch(torch.tensor(images), qadence_circuit, patch_size=6, n_atoms=n_atoms)
    print(output_embeddings.shape)  
    