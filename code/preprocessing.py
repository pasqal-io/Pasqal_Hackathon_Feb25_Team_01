import os
import numpy as np
import pydicom
import cv2
from PIL import Image

dicom_dir = "/home/eflammere/Pasqal_Hackathon_Feb25_Team_01/data/dicom"
output_dir = "data/png"

os.makedirs(output_dir, exist_ok=True)

patient_id = 1

for patient_folder in sorted(os.listdir(dicom_dir)):
    patient_path = os.path.join(dicom_dir, patient_folder)
    
    if os.path.isdir(patient_path):
        for file in os.listdir(patient_path):
            if file.endswith(".dcm"):
                dicom_path = os.path.join(patient_path, file)
                
                try:
                    dicom_data = pydicom.dcmread(dicom_path)
                    image = dicom_data.pixel_array

                    image = image.astype(np.float32)
                    image = (image - image.min()) / (image.max() - image.min()) * 255.0
                    image = image.astype(np.uint8)

                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    image = clahe.apply(image)

                    image_pil = Image.fromarray(image)
                    output_path = os.path.join(output_dir, f"{patient_id}.png")
                    image_pil.save(output_path)

                    print(f"Saved: {output_path}")
                    patient_id += 1

                except Exception as e:
                    print(f"Error processing {dicom_path}: {e}")
