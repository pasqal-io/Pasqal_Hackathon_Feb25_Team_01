import os
import requests
import pydicom
import zipfile
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class DICOMHandler:
    def __init__(self, api_url):
        self.api_url = api_url
    
    def load_images(self, series_uid, output_folder):
        ensure_directory(output_folder)
        response = requests.get(f"{self.api_url}getImage?SeriesInstanceUID={series_uid}", stream=True)
        content_type = response.headers.get("Content-Type", "").lower()
        
        if "zip" in content_type or zipfile.is_zipfile(BytesIO(response.content)):
            dicom_zip_path = "dicom_images.zip"
            with open(dicom_zip_path, "wb") as f:
                f.write(response.content)
            
            with zipfile.ZipFile(dicom_zip_path, "r") as zip_ref:
                zip_ref.extractall(output_folder)
            os.remove("dicom_images.zip")
        elif "octet-stream" in content_type:
            dicom_path = os.path.join(output_folder, "single_dicom.dcm")
            with open(dicom_path, "wb") as f:
                f.write(response.content)
        else:
            print("Unexpected content type. Response might not be a DICOM file.")
            print(response.text[:500])
    
    def save_as_png(self, dicom_folder, output_folder):
        ensure_directory(output_folder)
        dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith(".dcm")]
        
        for dicom_file in dicom_files:
            dicom_path = os.path.join(dicom_folder, dicom_file)
            dicom_data = pydicom.dcmread(dicom_path)
            
            # Normalize pixel values to [0, 255]
            pixel_array = dicom_data.pixel_array.astype(np.float32)
            pixel_array -= pixel_array.min()
            pixel_array /= (pixel_array.max() + 1e-8)
            pixel_array *= 255.0
            pixel_array = pixel_array.astype(np.uint8)
            
            # Save as PNG
            image = Image.fromarray(pixel_array)
            image.save(os.path.join(output_folder, dicom_file.replace(".dcm", ".png")))
    
    def display_images(self, dicom_folder):
        dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith(".dcm")]
        
        if not dicom_files:
            print("No DICOM files found in the folder.")
            return
        
        for dicom_file in dicom_files:
            dicom_path = os.path.join(dicom_folder, dicom_file)
            dicom_data = pydicom.dcmread(dicom_path)
            
            plt.figure(figsize=(6, 6))
            plt.imshow(dicom_data.pixel_array, cmap="gray")
            plt.axis("off")
            plt.title(f"DICOM: {dicom_file}")
            
            # Wait for key press to proceed, exit if any key is pressed
            plt.show(block=False)
            print("Press any key to continue to the next image, or close the window to exit.")
            if plt.waitforbuttonpress():
                plt.close()
                break
            
            plt.close()

if __name__ == "__main__":
    # API and uid of liver data from Cancer Imaging Archive
    API_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
    series_uid = "1.3.6.1.4.1.14519.5.2.1.1706.8374.139683127466268036038326476970"

    # data paths
    dicom_folder_path = "data/liver_dicom"
    png_folder_dath = "data/liver_png"

    dicom_handler = DICOMHandler(API_URL)
    dicom_handler.load_images(series_uid, dicom_folder_path)
    dicom_handler.display_images(dicom_folder_path)
    dicom_handler.save_as_png(dicom_folder_path, png_folder_dath)