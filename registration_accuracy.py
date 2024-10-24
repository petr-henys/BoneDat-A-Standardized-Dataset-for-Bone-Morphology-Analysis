from morpho_mapper import warp_inv  # Custom module for inverse warping operations
import os  # For file/directory operations
import pyvista as pv  # For 3D scientific visualization
from tqdm import trange  # For progress bar visualization
import numpy as np  # For numerical operations

if __name__ == '__main__':
    # Define path to the database containing synthetic patient data
    database_path = '/mnt/database/synthetic_patients/'
    
    # Get list of all patient directories in the raw data folder
    patients_list = list(os.scandir(os.path.join(database_path, 'raw')))
    
    # Load the template pelvic mesh
    mesh = pv.read('templates/geometry/base/pelvic.vtk')

    errors = []  # Initialize list to store registration errors
    # Iterate through each patient's data with a progress bar
    for i in trange(len(patients_list)):
        patient_id = patients_list[i].name  # Get current patient's ID
        # Construct paths to relevant directories and files
        patient_folder = os.path.join(database_path, 'raw', patient_id)
        registration_folder = os.path.join(database_path, 'registrations', patient_id)

        # Define paths to registration-related files
        warp_inv_file = os.path.join(registration_folder, 'warp_inv.nii.gz')  # Inverse warp field
        warp_fwd_file = os.path.join(registration_folder, 'warp_fwd.nii.gz')  # Forward warp field
        afn_fwd_file = os.path.join(registration_folder, 'afn_fwd.mat')  # Affine transformation
        u_ref_file = os.path.join(patient_folder, 'ref_deformation.npy')  # Reference deformation
        X_est_file = os.path.join(patient_folder, 'est_deformation.npy')  # Estimated deformation

        # Load the reference and estimated deformation fields
        u_ref = np.load(u_ref_file)
        X_est = np.load(X_est_file)
        
        # Apply inverse warp to mesh points
        est_X = warp_inv(mesh.points, warp_inv_file, afn_fwd_file)

        # Calculate the estimated deformation field
        u_est = est_X - mesh.points
        # Calculate and store the error between reference and estimated deformation
        errors.append(u_ref-u_est)

    # Convert errors list to numpy array
    errors = np.asarray(errors)
    # Save the raw registration errors
    np.save('additional_data/registration_errors.npy', errors)

    # Add statistical measures to the mesh
    mesh['mean'] = np.mean(errors, axis=0)  # Mean error at each point
    mesh['std'] = np.std(errors, axis=0)    # Standard deviation of errors at each point

    # Save the mesh with error statistics
    mesh.save('additional_data/registration_errors.vtk')