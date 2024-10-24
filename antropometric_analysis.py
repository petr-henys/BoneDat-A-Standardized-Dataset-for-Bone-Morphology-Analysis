import pandas as pd
import numpy as np
from pathlib import Path
import database as db  # Assuming this is a custom module for database interaction
from morpho_mapper import warp_fwd  # Assuming this is a custom module for image warping
from tqdm import tqdm  # For progress bars
from scipy.stats import mannwhitneyu  # For statistical testing

def collect_points2(palpation):
    """
    Collects and transforms 2D landmark points from a palpation dataset.

    Args:
        palpation (pd.DataFrame): DataFrame containing palpation data with columns 
                                   'x1', 'y1', 'z1', 'x2', 'y2', 'z2' for 
                                   two sets of landmark coordinates.

    Returns:
        np.ndarray: A NumPy array of shape (n_patients, n_landmarks, 3) containing 
                    the transformed 3D coordinates of the landmarks for each patient.
    """
    # Create DataFrames for the first and second sets of points
    p1 = pd.DataFrame({'x': palpation.x1, 'y': palpation.y1, 'z': palpation.z1})
    p2 = pd.DataFrame({'x': palpation.x2, 'y': palpation.y2, 'z': palpation.z2})
    
    # Concatenate the points into a single DataFrame
    points = pd.concat([p1, p2], axis=0, ignore_index=True)

    points_all = []
    # Iterate over patient IDs and apply transformations
    for patient_id in tqdm(patient_info, desc="Processing Patients"):  # Using tqdm for a progress bar
        patient_folder = database_registration / patient_id
        aff_file_fwd = patient_folder / 'afn_fwd.mat'  # Path to affine transformation file
        warp_file_fwd = patient_folder / 'warp_fwd.nii.gz'  # Path to warp field file
        
        # Apply the warp_fwd function (from morpho_mapper module) to transform the points
        points_trn = warp_fwd(points.to_numpy(), str(warp_file_fwd), str(aff_file_fwd))
        points_all.append(points_trn)

    return np.asarray(points_all)

def angle_between_3points(p1, p2, p3):
  """
  Calculates the angle formed by three 3D points (vectorized).

  Args:
    p1: A NumPy array of shape (n, 3) containing n first points.
    p2: A NumPy array of shape (n, 3) containing n second points (vertices).
    p3: A NumPy array of shape (n, 3) containing n third points.

  Returns:
    A NumPy array of shape (n,) containing the angles in degrees.
  """
  v1 = p1 - p2  # Vector from p2 to p1
  v2 = p3 - p2  # Vector from p2 to p3
  # Calculate the cosine of the angle using the dot product formula
  cosine_angle = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
  angle_radians = np.arccos(cosine_angle)  # Calculate the angle in radians
  angle_degrees = np.degrees(angle_radians)  # Convert to degrees
  return angle_degrees

def collect_points3(palpation):
    """
    Collects and transforms 3D landmark points from a palpation dataset.

    Args:
        palpation (pd.DataFrame): DataFrame containing palpation data with columns 
                                   'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3' 
                                   for three sets of landmark coordinates.

    Returns:
        np.ndarray: A NumPy array of shape (n_patients, n_landmarks, 3) containing 
                    the transformed 3D coordinates of the landmarks for each patient.
    """
    # Create DataFrames for the first, second, and third sets of points
    p1 = pd.DataFrame({'x': palpation.x1, 'y': palpation.y1, 'z': palpation.z1})
    p2 = pd.DataFrame({'x': palpation.x2, 'y': palpation.y2, 'z': palpation.z2})
    p3 = pd.DataFrame({'x': palpation.x3, 'y': palpation.y3, 'z': palpation.z3})
    
    # Concatenate the points into a single DataFrame
    points = pd.concat([p1, p2, p3], axis=0, ignore_index=True)

    points_all = []
    # Iterate over patient IDs and apply transformations
    for patient_id in tqdm(patient_info, desc="Processing Patients"):  # Using tqdm for a progress bar
        patient_folder = database_registration / patient_id
        aff_file_fwd = patient_folder / 'afn_fwd.mat'  # Path to affine transformation file
        warp_file_fwd = patient_folder / 'warp_fwd.nii.gz'  # Path to warp field file
        
        # Apply the warp_fwd function (from morpho_mapper module) to transform the points
        points_trn = warp_fwd(points.to_numpy(), str(warp_file_fwd), str(aff_file_fwd))
        points_all.append(points_trn)

    return np.asarray(points_all)

def compute_distances(points):
    """
    Computes distances between specific landmark pairs.

    Args:
        points (np.ndarray): A NumPy array of shape (n_patients, n_landmarks, 3) 
                             containing the 3D coordinates of the landmarks.

    Returns:
        dict: A dictionary where keys are distance names (e.g., 'anatomical conjugate') 
              and values are NumPy arrays of shape (n_patients,) containing the 
              computed distances for each patient.
    """
    p1 = points[:, 0:17]  # First set of landmarks
    p2 = points[:, 17:]  # Second set of landmarks

    # Calculate distances between specific landmark pairs
    dists = {
        'anatomical conjugate' : np.linalg.norm((p1[:, 0] + p1[:, 0]) / 2 - (p2[:, 0] + p2[:, 0]) / 2, axis=1),
        'true conjugate' : np.linalg.norm((p1[:, 1] + p1[:, 1]) / 2 - (p2[:, 1] + p2[:, 1]) / 2, axis=1),
        'diagonal conjugate' : np.linalg.norm((p1[:, 2] + p1[:, 2]) / 2 - (p2[:, 2] + p2[:, 3]) / 2, axis=1),
        'anatomical transverse' : np.linalg.norm((p1[:, 3] + p1[:, 4]) / 2 - (p2[:, 3] + p2[:, 4]) / 2, axis=1),
        'left oblique' : np.linalg.norm((p1[:, 5] + p1[:, 5]) / 2 - (p2[:, 5] + p2[:, 5]) / 2, axis=1),
        'right oblique' : np.linalg.norm((p1[:, 6] + p1[:, 6]) / 2 - (p2[:, 6] + p2[:, 6]) / 2, axis=1),
        'straight conjugate' : np.linalg.norm((p1[:, 7] + p1[:, 8]) / 2 - (p2[:, 7] + p2[:, 8]) / 2, axis=1),
        'median conjugate' : np.linalg.norm((p1[:, 9] + p1[:, 10]) / 2 - (p2[:, 9] + p2[:, 10]) / 2, axis=1),
        'bis-ischiadic' : np.linalg.norm((p1[:, 11] + p1[:, 11]) / 2 - (p2[:, 11] + p2[:, 11]) / 2, axis=1),
        'pubic tubercle height' : np.linalg.norm((p1[:, 12] + p1[:, 12]) / 2 - (p2[:, 12] + p2[:, 12]) / 2, axis=1),
        'promontory to coccyx' : np.linalg.norm((p1[:, 13] + p1[:, 13]) / 2 - (p2[:, 13] + p2[:, 13]) / 2, axis=1),
        'sacrum-S3/S4' : np.linalg.norm((p1[:, 14] + p1[:, 14]) / 2 - (p2[:, 14] + p2[:, 14]) / 2, axis=1),
        'S3/S4-coccyx' : np.linalg.norm((p1[:, 15] + p1[:, 15]) / 2 - (p2[:, 15] + p2[:, 15]) / 2, axis=1),
        'ischialspines' : np.linalg.norm((p1[:, 16] + p1[:, 16]) / 2 - (p2[:, 16] + p2[:, 16]) / 2, axis=1),
    }

    return dists

def compute_angles(points):
    """
    Computes angles between specific landmark triplets.

    Args:
        points (np.ndarray): A NumPy array of shape (n_patients, n_landmarks, 3) 
                             containing the 3D coordinates of the landmarks.

    Returns:
        dict: A dictionary where keys are angle names (e.g., 'the I-P-C angle') 
              and values are NumPy arrays of shape (n_patients,) containing the 
              computed angles for each patient.
    """
    p1 = points[:, 0:4]  # First set of landmarks
    p2 = points[:, 4:8]  # Second set of landmarks
    p3 = points[:, 8:]  # Third set of landmarks

    # Calculate angles between specific landmark triplets using the angle_between_3points function
    angles = {
        'the I-P-C angle': angle_between_3points((p1[:, 0] + p1[:, 1]) / 2, 
                                                         (p2[:, 0] + p2[:, 1]) / 2, 
                                                         (p3[:, 0] + p3[:, 1]) / 2),
        'the P-C-O angle': angle_between_3points(p1[:, 2], p2[:, 2], p3[:, 2]),
        'the angle at S3': angle_between_3points(p1[:, 3], p2[:, 3], p3[:, 3]),
    }

    return angles

if __name__ == '__main__':
    # Define paths to database directories
    database_directory = Path('/mnt/database/BoneDat')
    database_raw = database_directory / 'raw'
    database_registration = database_directory / 'derived' / 'registrations'
    database_fields = database_directory / 'derived' / 'fields'
    patient_info = db.collect_patient_info(database_raw)

    distance_points = pd.read_excel('additional_data/ref_distances.xlsx')
    angle_points = pd.read_excel('additional_data/ref_angles.xlsx')

    #points = collect_points2(distance_points) # uncomment to recompute
    #np.save('additional_data/points4distances.npy', points)
    points4dists = np.load('additional_data/points4distances.npy')

    #points = collect_points3(angle_points) # uncomment to recompute
    #np.save('additional_data/points4angles.npy', points)
    points4angles = np.load('additional_data/points4angles.npy')

    dists = compute_distances(points4dists)
    angles = compute_angles(points4angles)

    pelvis_file_name = 'templates/geometry/base/pelvic.vtk'

    sex = db.get_as_numpy(patient_info, 'sex')
    age = db.get_as_numpy(patient_info, 'age')

    # --- STATISTICAL TESTING (MANN-WHITNEY U TEST) ---
    # Prepare lists to store the results
    distance_results = []
    angle_results = []

    # Perform Mann-Whitney U test for sex differences in distances
    for diam_name in dists.keys():
        AC_F = dists[diam_name][sex == 'F']  # Distances for females
        AC_M = dists[diam_name][sex == 'M']  # Distances for males
        m, p = mannwhitneyu(AC_F, AC_M)  # Perform the test

        # Append results to the list
        distance_results.append([diam_name, p, AC_F.mean(), AC_F.std(), AC_M.mean(), AC_M.std()])

    # Perform Mann-Whitney U test for sex differences in angles
    for angle_name in angles.keys():
        AC_F = angles[angle_name][sex == 'F']  # Angles for females
        AC_M = angles[angle_name][sex == 'M']  # Angles for males
        m, p = mannwhitneyu(AC_F, AC_M)  # Perform the test

        # Append results to the list
        angle_results.append([angle_name, p, AC_F.mean(), AC_F.std(), AC_M.mean(), AC_M.std()])


    # --- SAVE RESULTS TO EXCEL ---
    # Create DataFrames from the results lists
    df_distances = pd.DataFrame(distance_results, columns=['Distance', 'p-value', 'Female Mean', 'Female SD', 'Male Mean', 'Male SD'])
    df_angles = pd.DataFrame(angle_results, columns=['Angle', 'p-value', 'Female Mean', 'Female SD', 'Male Mean', 'Male SD'])

    # Save DataFrames to an Excel file
    with pd.ExcelWriter('additional_data/anthropometric_results.xlsx') as writer:
        df_distances.to_excel(writer, sheet_name='Distances', index=False)
        df_angles.to_excel(writer, sheet_name='Angles', index=False)


