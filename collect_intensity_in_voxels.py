import numpy as np
from pathlib import Path
import ants
import database as db
from tqdm import tqdm
from utils import shift_intensity

if __name__ == '__main__':

    # Define directories for the database, raw data, registrations, and derived fields
    database_directory = Path('/mnt/database/BoneDat')
    database_raw = database_directory / 'raw'
    database_registration = database_directory / 'derived' / 'registrations'
    database_fields = database_directory / 'derived' / 'fields'

    # Load the template image used for registration
    template = ants.image_read('templates/segmentation/template.nii.gz')

    # Collect patient information from the database
    patient_info = db.collect_patient_info(database_raw)

    # Initialize lists to store processed images and potentially full images
    images = []
    images_full = []  

    # Create a mask from the template to extract relevant regions
    mask = template > 1e-2  

    # Loop through each patient in the dataset
    for patient_id in tqdm(patient_info, desc="Processing Patients"):
        # Define paths to registration files for the current patient
        patient_folder = database_registration / patient_id
        aff_file_fwd = patient_folder / 'afn_fwd.mat'  # Affine transformation
        warp_file_fwd = patient_folder / 'warp_fwd.nii.gz'  # Forward warp field
        warp_file_inv = patient_folder / 'warp_inv.nii.gz'  # Inverse warp field
        warped_image_file = patient_folder / 'warped.nii.gz'  # Warped image

        # Load the warped image
        warped_image = ants.image_read(str(warped_image_file))

        # Normalize image intensity by shifting the minimum value to zero
        warped_image = shift_intensity(warped_image, -warped_image.min())  

        # Extract the region of interest using the mask and add it to the list
        images.append(warped_image[mask])  

    # Convert the list of processed images to a NumPy array
    images = np.asarray(images)  

    # Save the processed image data as a NumPy array file
    np.save('additional_data/I_voxels.npy', images)