import pandas as pd
from pathlib import Path
import numpy as np

def collect_patient_info(root_directory, metadata_file='metadata.xlsx'):
    
    """
    Collects patient sex and age from metadata files in patient directories.

    Args:
        root_directory (str): Path to the directory containing patient folders.
        metadata_file (str, optional): Name of the metadata file. 
                                     Defaults to 'metadata.xlsx'.

    Returns:
        dict: A dictionary mapping patient IDs to dictionaries containing 
               sex and age.
    """
    patient_data = {}  # Dictionary to store results

    for patient_folder in Path(root_directory).iterdir():
        if patient_folder.is_dir():
            metadata_path = patient_folder / metadata_file

            if metadata_path.exists():
                metadata = pd.read_excel(metadata_path)
                try:
                    patient_id = patient_folder.name  # Use folder name as ID
                    sex = metadata['sex'][0]  # Assuming single-row metadata
                    age = metadata['CT date'][0] - metadata['born'][0]
                except KeyError:
                    print(f"Warning: Missing 'Sex' or 'Age' data in {metadata_path}")
                    continue  # Skip this patient if data is incomplete
                except IndexError:
                    print(f"Warning: Empty metadata file in {metadata_path}")
                    continue
                patient_data[patient_id] = {'sex': sex, 'age': age}

    return patient_data

def get_as_numpy(patient_info, item):
    """
    Extracts a specific data item from the patient_info dictionary 
    and returns it as a NumPy array.

    Args:
        patient_info (dict): The dictionary containing patient data.
        item (str): The key of the item to extract.

    Returns:
        np.ndarray: A NumPy array containing the extracted data.
    """
    
    return np.array([info[item] for info in patient_info.values()])

def get_patientid_as_numpy(patient_info):
    """
    Extracts patient IDs from the patient_info dictionary and 
    returns them as a NumPy array.

    Args:
        patient_info (dict): The dictionary containing patient data.

    Returns:
        np.ndarray: A NumPy array containing the patient IDs.
    """
    return np.array(list(patient_info.keys()))

if __name__ == '__main__':
    database_directory = '/mnt/database/BoneDat/raw'
    patient_info = collect_patient_info(database_directory)
    sex = get_as_numpy(patient_info, 'sex')
    age = get_as_numpy(patient_info, 'age')
    patient_ids = get_patientid_as_numpy(patient_info)