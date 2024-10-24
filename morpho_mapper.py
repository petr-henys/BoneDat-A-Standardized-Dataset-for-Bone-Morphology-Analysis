import pyvista as pv
import numpy as np
from pathlib import Path
import ants
import database as db
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

def warp_fwd(points, warp_file_fwd, aff_file_fwd):
    """
    Apply forward warping transformation to a set of points using ANTs.
    
    Args:
        points (numpy.ndarray): Array of 3D points to transform (N x 3)
        warp_file_fwd (str): Path to the forward warp transformation file (.nii.gz)
        aff_file_fwd (str): Path to the forward affine transformation file (.mat)
    
    Returns:
        numpy.ndarray: Transformed points in the target space
    """
    points = pd.DataFrame(points, columns=['x', 'y', 'z'])
    return ants.apply_transforms_to_points(dim=3, points=points,
                             transformlist=[warp_file_fwd, aff_file_fwd], 
                             whichtoinvert=[False, False]).values

def warp_inv(points, warp_file_inv, aff_file_fwd):
    """
    Apply inverse warping transformation to a set of points using ANTs.
    
    Args:
        points (numpy.ndarray): Array of 3D points to transform (N x 3)
        warp_file_inv (str): Path to the inverse warp transformation file (.nii.gz)
        aff_file_fwd (str): Path to the forward affine transformation file (.mat)
    
    Returns:
        numpy.ndarray: Inversely transformed points in the original space
    """
    points = pd.DataFrame(points, columns=['x', 'y', 'z'])
    return ants.apply_transforms_to_points(dim=3, points=points,
                             transformlist=[aff_file_fwd, warp_file_inv], 
                             whichtoinvert=[True, False]).values

def collect_deformations(patient_info, database_registration, mesh):
    """
    Collect forward and inverse deformation fields for a set of patients.
    
    Args:
        patient_info (list): List of patient IDs
        database_registration (Path): Path to the registration database directory
        mesh (pyvista.UnstructuredGrid): Reference mesh for deformation
    
    Returns:
        numpy.ndarray: Array of shape (N_patients, 2, N_points, 3) containing
                      forward and inverse transformations for each patient
    """
    mesh_points = mesh.points.astype(np.float32)  # Reduce precision if acceptable

    def process_patient(patient_id):
        """Helper function to process individual patient transformations"""
        patient_folder = database_registration / patient_id
        aff_file_fwd = patient_folder / 'afn_fwd.mat'
        warp_file_fwd = patient_folder / 'warp_fwd.nii.gz'
        warp_file_inv = patient_folder / 'warp_inv.nii.gz'
        
        return (warp_fwd(mesh_points, str(warp_file_fwd), str(aff_file_fwd)),
                warp_inv(mesh_points, str(warp_file_inv), str(aff_file_fwd)))

    x_trns_array = []
    for patient_id in tqdm(patient_info, desc="Processing Patients"):
        x_trns_array.append(process_patient(patient_id))
    
    return np.asarray(x_trns_array)

def compute_tetrahedron_centroids(mesh):
    """
    Compute centroids of tetrahedra in a tetrahedral mesh.
    
    Args:
        mesh (pyvista.UnstructuredGrid): Tetrahedral mesh
    
    Returns:
        numpy.ndarray: Array of centroid coordinates for each tetrahedron
    """
    cells = mesh.cells.reshape((-1, 5))[:, 1:]  # Extract vertex indices for each tetrahedron
    points = mesh.points  # Vertex coordinates
    centroids = np.mean(points[cells], axis=1)
    return centroids

def idw(image, mesh, threshold=0., power=2, k_neighbors=8, method='nodes'):
    """
    Perform Inverse Distance Weighting interpolation from image voxels to mesh points.
    
    Args:
        image (ants.core.ants_image.ANTsImage): Input image
        mesh (pyvista.UnstructuredGrid): Target mesh
        threshold (float): Minimum intensity threshold for valid voxels
        power (int): Power parameter for distance weighting
        k_neighbors (int): Number of nearest neighbors to consider
        method (str): Interpolation method ('nodes' or 'centroids')
    
    Returns:
        numpy.ndarray: Interpolated intensity values at mesh points/centroids
    """
    ct_array = image.numpy()
    origin = image.origin
    spacing = image.spacing
    
    # Select interpolation points based on method
    if method=='nodes':
        points = mesh.points
    elif method=='centroids':
        points = compute_tetrahedron_centroids(mesh)

    # Create 3D grid of voxel centers
    grid_x, grid_y, grid_z = np.mgrid[0:ct_array.shape[0], 0:ct_array.shape[1], 0:ct_array.shape[2]]
    voxel_centers = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T
    voxel_centers = voxel_centers * spacing + origin  # Convert indices to world coordinates

    voxel_values = ct_array.ravel()
    valid_mask = voxel_values >= threshold
    valid_voxel_centers = voxel_centers[valid_mask]
    valid_voxel_values = voxel_values[valid_mask]

    if len(valid_voxel_values) == 0:
        return np.zeros(len(points))

    # Create KD-tree for efficient nearest neighbor search
    tree = KDTree(valid_voxel_centers)
    distances, indices = tree.query(points, k=k_neighbors)
    distances[distances == 0] = 1e-6  # Prevent division by zero

    valid_voxel_neighbors = valid_voxel_values[indices]
    weights = 1.0 / distances**power
    weighted_sum = np.sum(weights * valid_voxel_neighbors, axis=1)
    sum_of_weights = np.sum(weights, axis=1)

    return weighted_sum / sum_of_weights

def collect_intensities(patient_info, database_registration, mesh):
    """
    Collect intensity values from warped images for all patients.
    
    Args:
        patient_info (list): List of patient IDs
        database_registration (Path): Path to the registration database directory
        mesh (pyvista.UnstructuredGrid): Reference mesh for intensity mapping
    
    Returns:
        numpy.ndarray: Array of shape (N_patients, N_points) containing
                      intensity values for each patient at mesh points
    """
    def process_patient(patient_id, mesh):
        """Helper function to process individual patient intensities"""
        patient_folder = database_registration / patient_id
        warped_img = patient_folder / 'lumbopelvic_warped.nii.gz'
        return idw(ants.image_read(str(warped_img)), mesh, 
                  threshold=0.05, k_neighbors=4,
                  method='nodes')

    i_array = []
    for patient_id in tqdm(patient_info, desc="Processing Patients"):
        i_array.append(process_patient(patient_id, mesh))
    
    return np.asarray(i_array)

if __name__ == '__main__':
    database_directory = Path('/mnt/database/BoneDat')
    database_raw = database_directory / 'raw'
    database_registration = database_directory / 'derived' / 'registrations'
    database_fields = database_directory / 'derived' / 'fields'

    patient_info = db.collect_patient_info(database_raw)
    pelvis_file_name = 'templates/geometry/base/pelvic.vtk'

    # Process and save data
    mesh = pv.read(pelvis_file_name)
    X = collect_deformations(patient_info, database_registration, mesh)
    np.save('additional_data/X_fwd.npy', X[:, 0])
    np.save('additional_data/X_inv.npy', X[:, 1])
    I = collect_intensities(patient_info, database_registration, mesh)
    np.save('additional_data/I.npy', I)