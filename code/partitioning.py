import os
import numpy as np

folder_path = "/home/eflammere/Pasqal_Hackathon_Feb25_Team_01/embeddings/train"
output_file = "train_image_embeddings.npy"

npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

merged_list = []
for file in npy_files:
    data = np.load(os.path.join(folder_path, file))

    if data.shape != (6, 6, 128):
        print(f"Skipping {file} due to unexpected shape: {data.shape}")
        continue

    merged_list.append(data)

if merged_list:
    merged_array = np.stack(merged_list)
    np.save(os.path.join(folder_path, output_file), merged_array)
    print(f"Saved merged file: {output_file}, shape: {merged_array.shape}")
else:
    print("No valid .npy files found!")
