# Import necessary libraries
from utils import shift_intensity, create_dir  # Custom functions for intensity shifting and directory creation
import ants  # Advanced Normalization Tools (ANTs) library for image processing
from tqdm import tqdm  # Progress bar library for visualization
import tempfile  # Temporary file handling library
import database as db  # Custom database module for data management
from pathlib import Path  

# Main execution block
if __name__ == '__main__':
    # Define paths to database directories
    database_directory = Path('/mnt/database/BoneDat')  # Root directory containing all data
    database_raw = database_directory / 'raw'  # Directory containing original, unprocessed images
    database_registration = database_directory / 'derived' / 'registrations'  # Directory for registration outputs
    
    # Collect patient information from the database
    # This includes metadata and file paths for all patients in the study
    patient_info = db.collect_patient_info(database_raw)

    # Load and preprocess the initial template image
    # This template serves as the starting point for the iterative template building process
    initial_template = ants.image_read(f'templates/segmentation/template.nii.gz')
    
    # Preprocessing steps for the initial template:
    # 1. Shift intensity values to ensure non-negative values
    initial_template = shift_intensity(initial_template, -initial_template.numpy().min())
    
    # 2. Resample to a standardized resolution (0.8mm isotropic)
    initial_template = ants.resample_image(initial_template, [0.8, 0.8, 0.8])
    
    # 3. Normalize intensity values to a standard range
    initial_template = ants.iMath_normalize(initial_template)

    # Set up temporary directory for processing
    # This directory will store intermediate files during template construction
    temp_dir = '/mnt/pracovni/tmp'
    create_dir(temp_dir)  # Ensure the directory exists
    tempfile.tempdir = temp_dir  # Configure tempfile module to use this directory

    # Collect and preprocess patient images
    images = []  # List to store preprocessed patient images
    for patient_path in tqdm(patient_info, desc='collecting images...'):
        patient_folder = database_registration / patient_path 

        # Load rigidly aligned image
        # Note: SyN registration requires pre-aligned images
        mi = ants.image_read(str(patient_folder / 'rigid.nii.gz'))
        
        # Preprocess each patient image:
        # 1. Calculate and remove intensity offset
        int_offset = mi.numpy().min()
        mi = shift_intensity(mi, -int_offset)
        
        # 2. Normalize intensity values
        mi = ants.iMath_normalize(mi)
    
        images.append(mi)

    # Build new template using ANTs
    # This is the main template construction step using symmetric normalization
    new_template = ants.build_template(
        initial_template=initial_template,  # Starting point for template construction
        image_list=images,  # List of preprocessed patient images
        
        # Registration parameters:
        type_of_transform='SyN',  # Symmetric Normalization transformation
        iterations=3,  # Number of template building iterations
        
        # Similarity metrics for registration:
        aff_metric='mattes',  # Mattes Mutual Information for affine registration
        syn_metric='mattes',  # Mattes Mutual Information for deformable registration
        
        # Sampling parameters:
        syn_sampling=32,  # Number of samples for SyN optimization
        aff_sampling=32,  # Number of samples for affine optimization
        
        # Multi-resolution parameters:
        aff_iterations=[2100, 1200, 1200, 100],  # Iterations at each resolution level for affine
        reg_iterations=[100, 80, 50, 30]  # Iterations at each resolution level for SyN
    )

    # Post-process the final template
    new_template = ants.iMath_normalize(new_template)  # Normalize the final template
    
    # Save the resulting template
    ants.image_write(new_template, 'templates/new_template.mha')