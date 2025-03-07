import os
import shutil
import pydicom

def find_dicom_file(directory):
    """Recursively search for a DICOM file in the given directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".dcm"):
                return os.path.join(root, file)
    return None

def move_dicom_files(source_root, dest_root):
    for i in range(1, 106):
        # Format the directory name with leading zeros
        source_dir = os.path.join(source_root, f"HCC_{i:03d}")  # Using 3 digits with leading zeros
        dest_dir = os.path.join(dest_root, f"HCC_{i:03d}")  # Using 3 digits with leading zeros
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        if os.path.exists(source_dir):
            dicom_file = find_dicom_file(source_dir)
            
            if dicom_file:
                dicom_filename = os.path.basename(dicom_file)
                dest_path = os.path.join(dest_dir, dicom_filename)
                
                shutil.move(dicom_file, dest_path)
                print(f"Moved {dicom_filename} from {source_dir} to {dest_dir}")
            else:
                print(f"No DICOM file found in {source_dir}")
        else:
            print(f"Source directory does not exist: {source_dir}")

if __name__ == "__main__":
    source_root = "/mnt/c/Users/Win10/Desktop/manifest-1643035385102/HCC-TACE-Seg"  # Update with actual path
    dest_root = "/home/eflammere/Pasqal_Hackathon_Feb25_Team_01/data/dicom"  # Update with actual path
    move_dicom_files(source_root, dest_root)
