import pyvista as pv
import numpy as np
from pathlib import Path
import database as db
from tqdm import tqdm
from morpho_mapper import warp_fwd
import meshio, pygmsh

def orient_mesh(v, t):
    """
    Ensures proper orientation of tetrahedral elements in a mesh by checking and correcting
    negative volume elements.
    
    Args:
        v (numpy.ndarray): Vertex coordinates array of shape (n_vertices, 3)
        t (numpy.ndarray): Tetrahedral elements array of shape (n_elements, 4)
    
    Returns:
        numpy.ndarray: Properly oriented tetrahedral elements array
    
    Notes:
        - Computes tetrahedra volumes using cross products
        - Flips negative volume tetrahedra by swapping vertices
        - Prints number of corrected elements
    """
    # Extract vertices for each tetrahedron
    t0, t1, t2, t3 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    v0 = v[t0, :]
    v1 = v[t1, :]
    v2 = v[t2, :]
    v3 = v[t3, :]
    
    # Compute edge vectors
    e0 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0
    
    # Compute signed volumes
    cr = np.cross(e0, e2)
    vol = np.sum(e3 * cr, axis=1)
    
    # Identify and fix negative volume elements
    negtet = vol < 0.0
    negnum = np.sum(negtet)
    
    if negnum == 0:
        print("Mesh is oriented, nothing to do")
        return t
        
    # Correct orientation by swapping vertices
    tnew = t
    temp = t[negtet, 1]
    tnew[negtet, 1] = t[negtet, 2]
    tnew[negtet, 2] = temp
    onum = np.sum(negtet)
    print("Flipped " + str(onum) + " tetrahedra")
    return tnew

def optimise_tetrahedral_mesh(mesh, method='HighOrder'):
    """
    Optimizes a tetrahedral mesh using specified optimization method.
    
    Args:
        mesh (meshio.Mesh): Input tetrahedral mesh to optimize
        method (str, optional): Optimization method to use. Defaults to 'HighOrder'.
            Valid options are those supported by pygmsh.optimize()
    
    Returns:
        meshio.Mesh: Optimized tetrahedral mesh
    
    Notes:
        - First ensures proper element orientation
        - Then applies the specified optimization method using pygmsh
        - Returns original mesh if method is None
    """
    elements = mesh.cells_dict[mesh.celltypes[0]]
    nodes = mesh.points
    elements = orient_mesh(nodes, elements)
    mesh_opt = meshio.Mesh(nodes, [('tetra', elements)])
    
    if method is not None:
        mesh_opt = pygmsh.optimize(mesh_opt, 
                                 method=method,
                                 verbose=False)
    return mesh_opt

if __name__ == '__main__':
    # Define database directory structure
    database_directory = Path('/mnt/database/BoneDat')
    database_raw = database_directory / 'raw'
    database_registration = database_directory / 'derived' / 'registrations'
    database_fields = database_directory / 'derived' / 'fields'
    database_geometry = database_directory / 'derived' / 'geometries'

    # Load patient information and template mesh
    patient_info = db.collect_patient_info(database_raw)
    pelvis_file_name = 'templates/geometry/base/pelvic.vtk'
    template = pv.read(pelvis_file_name)

    # Process each patient
    for patient_id in tqdm(patient_info, desc="Processing Patients"):
        # Set up patient-specific paths
        patient_folder = database_registration / patient_id
        aff_file_fwd = patient_folder / 'afn_fwd.mat'
        warp_file_fwd = patient_folder / 'warp_fwd.nii.gz'
        
        # Transform template to patient-specific geometry
        warped_points = warp_fwd(template.points, str(warp_file_fwd), str(aff_file_fwd))
        warped_pelvis = template.copy()
        warped_pelvis.points = warped_points
        
        # Optimize and save mesh
        mesh = optimise_tetrahedral_mesh(warped_pelvis)
        mesh.write(database_geometry / patient_id / 'masked.vtk')
        
        # Reload and save in ASCII format
        mesh = pv.read(database_geometry / patient_id / 'masked.vtk')
        mesh.save(database_geometry / patient_id / 'masked.vtk', binary=False)