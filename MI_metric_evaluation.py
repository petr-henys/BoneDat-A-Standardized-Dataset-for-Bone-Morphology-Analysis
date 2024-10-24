import ants
from utils import shift_intensity  # Import function to adjust image intensity
from database import collect_patient_info  # Import function to gather patient data
from pathlib import Path  # For handling file paths
import pandas as pd  # For data manipulation and analysis
from tqdm import tqdm  # For displaying progress bars

if __name__ == '__main__':
    # Define directories for accessing data
    database_directory = Path('/mnt/database/BoneDat')
    database_raw = database_directory / 'raw'
    database_registration = database_directory / 'derived' / 'registrations'
    database_fields = database_directory / 'derived' / 'fields'

    # Collect patient information from the raw data directory
    patient_info = collect_patient_info(database_raw)

    # Load the template image and normalize its intensity
    template = ants.image_read('templates/segmentation/template.nii.gz')
    template = ants.iMath_normalize(template)
    
    # Define the image similarity metric (not used in this script, but kept for clarity)
    metric = 'JointHistogramMutualInformation'  
    
    results = []  # Initialize a list to store the results

    # Iterate over each patient ID, displaying a progress bar
    for patient_id in tqdm(patient_info, desc="Processing Patients"):  
        patient_folder = database_registration / patient_id  # Construct the patient's folder path
        
        # Load warped and rigid registered images
        warped = ants.image_read(str(patient_folder / 'warped.nii.gz'))  
        rigid = ants.image_read(str(patient_folder / 'rigid.nii.gz'))  

        # Shift intensity values to ensure non-negative values
        int_offset = warped.numpy().min()  
        warped = shift_intensity(warped, -int_offset)  

        int_offset = rigid.numpy().min()
        rigid = shift_intensity(rigid, -int_offset)

        # Normalize the intensity of warped and rigid images
        warped = ants.iMath_normalize(warped)  
        rigid = ants.iMath_normalize(rigid)  

        # Calculate mutual information between:
        # 1. Rigid image and itself (reference)
        mi_ref = ants.image_mutual_information(rigid, rigid)  
        # 2. Template and warped image
        mi_warped = ants.image_mutual_information(template, warped)  
        # 3. Template and rigid image
        mi_rigid = ants.image_mutual_information(template, rigid)  

        # Print the calculated mutual information values
        print(f'REF={mi_ref:.3f}, WARPED={mi_warped:.3f}, RIGID={mi_rigid:.3f}')  
        
        # Store the results in a dictionary
        results.append({"REF": mi_ref, "WARPED": mi_warped, "RIGID": mi_rigid})  

        # Delete variables to free up memory (optional, Python has garbage collection)
        del mi_ref  
        del mi_warped  
        del mi_rigid  

    # Create a Pandas DataFrame from the results
    df = pd.DataFrame(results)  
    # Save the DataFrame to an Excel file
    df.to_excel('additional_data/registration_metrics.xlsx')