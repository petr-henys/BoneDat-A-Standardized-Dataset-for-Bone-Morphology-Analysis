import pyvista as pv
import numpy as np
from pathlib import Path
import database as db  # Assuming this is a custom module for database interaction
from tqdm import tqdm  # For progress bar

def surface_mesh_quality(stl_file):
    """
    Analyzes the quality of a surface mesh provided as an STL file.

    This function computes various quality metrics for a triangular mesh, including:

    * Non-manifold edges:  Identifies edges shared by more than two faces, 
                            indicating inconsistencies in the mesh topology.
    * Degenerate faces: Counts faces with very small or zero area, which can 
                        cause problems in simulations or rendering.
    * Aspect ratio: Calculates the aspect ratio of each triangle (ratio of 
                    longest to shortest edge) to assess element quality.
    * Watertightness: Checks if the mesh is a closed manifold (no gaps or holes).
    * Curvature:  Estimates the mean curvature of the mesh, providing 
                information about surface smoothness (optional, may not always be reliable).

    Args:
        stl_file (str): Path to the STL file containing the surface mesh.

    Returns:
        dict: A dictionary containing the following quality metrics:
            - 'non_manifold_edges_count': Number of non-manifold edges.
            - 'degenerate_faces_count': Number of degenerate faces.
            - 'max_aspect_ratio': Maximum aspect ratio among all triangles.
            - 'min_aspect_ratio': Minimum aspect ratio.
            - 'is_watertight': Boolean indicating if the mesh is watertight.
            - 'max_curvature': Maximum mean curvature (optional, may be NaN).
            - 'min_curvature': Minimum mean curvature (optional, may be NaN).
    """

    # Load the STL file using pyvista
    mesh = pv.read(stl_file)

    # Initialize a dictionary to store the calculated quality metrics
    quality_metrics = {}

    # Check for non-manifold edges (edges shared by more than two faces)
    non_manifold_edges = mesh.extract_feature_edges(
        feature_edges=False, 
        boundary_edges=False, 
        manifold_edges=False, 
        non_manifold_edges=True
    )
    quality_metrics['non_manifold_edges_count'] = non_manifold_edges.n_cells

    # Check for degenerate faces (faces with very small area)
    cell_areas = mesh.compute_cell_sizes()['Area']
    degenerate_faces = cell_areas < 1e-12  # Define a threshold for degenerate faces
    quality_metrics['degenerate_faces_count'] = np.sum(degenerate_faces)

    # Compute the aspect ratio of each triangle in the mesh
    quality = mesh.compute_cell_quality(quality_measure='aspect_ratio')
    aspect_ratios = quality['CellQuality']
    quality_metrics['max_aspect_ratio'] = aspect_ratios.max()
    quality_metrics['min_aspect_ratio'] = aspect_ratios.min()

    # Check if the mesh is watertight (a closed manifold)
    quality_metrics['is_watertight'] = mesh.is_manifold

    # Optionally compute the mean curvature of the mesh
    try:
        curvature = mesh.curvature(curv_type='mean')  # Calculate mean curvature
        quality_metrics['max_curvature'] = curvature.max()
        quality_metrics['min_curvature'] = curvature.min()
    except:  # Catch any exceptions that might occur during curvature calculation
        quality_metrics['max_curvature'] = float('nan')  # Set to NaN if calculation fails
        quality_metrics['min_curvature'] = float('nan')

    return quality_metrics  # Return the dictionary of quality metrics

def volume_mesh_quality(mesh_file):
  """
  Calculates mesh quality indicators for a tetrahedral mesh.

  Args:
    mesh_file (str): Path to the mesh file.

  Returns:
    numpy.ndarray: An array of cell quality values (aspect ratios).
  """
  mesh = pv.read(mesh_file)  # Load the mesh file
  return mesh.compute_cell_quality(quality_measure='aspect_ratio')['CellQuality']  # Calculate and return aspect ratios


if __name__ == '__main__':
    database_directory = Path('/mnt/database/BoneDat')
    raw = database_directory / 'raw'  # Path to the raw data
    registration = database_directory / 'derived' / 'registrations' 
    fields = database_directory / 'derived' / 'fields'  
    geometry = database_directory / 'derived' / 'geometries' 

    patient_info = db.collect_patient_info(raw)  # Collect patient information from the database
    
    ref_quality = volume_mesh_quality('templates/geometry/base/pelvic.vtk')
    template = pv.read('templates/geometry/base/pelvic.vtk')  # Load the reference mesh

    measures = []  # Initialize a list to store mesh quality measures for each patient
    for patient_id in tqdm(patient_info, desc="Processing Patients"):
        metrics = volume_mesh_quality(geometry / patient_id / 'masked.vtk') 
        measures.append(metrics)  # Add the metrics to the list

    measures = np.asarray(measures)  # Convert the list of metrics to a NumPy array

    # Add the calculated mesh quality data to the reference mesh
    template['Mean jacobian'] = measures.mean(0)  # Mean Jacobian across all patients
    template['Std jacobian'] = measures.std(0)  # Standard deviation of Jacobian
    template['Template jacobian'] = ref_quality  # Jacobian of the reference mesh
    template.save('additional_data/mesh_quality.vtk')  # Save the updated mesh with quality data