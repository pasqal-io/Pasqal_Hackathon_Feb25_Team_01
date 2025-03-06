import os
import pandas as pd
import pydicom
import numpy as np
import cv2
from tqdm import tqdm

def dicom_to_png(input_folder, output_folder, csv_file):
    benign_dir = os.path.join(output_folder, "0")
    malign_dir = os.path.join(output_folder, "1")
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malign_dir, exist_ok=True)
    
    labels_df = pd.read_csv(csv_file)
    labels_dict = dict(zip(labels_df['filename'], labels_df['label'])) # Column name to be checked
    
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".dcm"):
            dicom_path = os.path.join(input_folder, filename)
            ds = pydicom.dcmread(dicom_path)
            pixel_array = ds.pixel_array
            
            pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
            pixel_array = pixel_array.astype(np.uint8)
            
            base_filename = os.path.splitext(filename)[0]
            
            label = labels_dict.get(base_filename, None)
            if label is not None:
                output_path = os.path.join(benign_dir if label == 0 else malign_dir, base_filename + ".png")
                cv2.imwrite(output_path, pixel_array)


# Path to dicom, path to output folder and path to .csv
dicom_to_png("", "", "")