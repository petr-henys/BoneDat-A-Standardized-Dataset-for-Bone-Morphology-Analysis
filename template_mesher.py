import SimpleITK as sitk
import numpy as np
from skimage import measure
import pyvista as pv
import pyacvd
import os
import trimesh
import gmsh
import pygmsh

def convert_stl_to_ascii(stl_file_name, output_file_name):
    """
    Convert a binary STL file to ASCII format.

    Args:
        stl_file_name (str): Path to the input STL file
        output_file_name (str): Path where the ASCII STL file will be saved

    Returns:
        str: Path to the converted ASCII file

    Raises:
        FileNotFoundError: If the input STL file doesn't exist
    """
    mesh = trimesh.load_mesh(stl_file_name)
    mesh.export(output_file_name, file_type='stl')
    return output_file_name

def estimate_surface_element_size(stl_file_name):
    """
    Estimate the appropriate element size for mesh generation based on the input STL file.

    Args:
        stl_file_name (str): Path to the STL file

    Returns:
        float: Average edge length of the triangles in the mesh

    Raises:
        FileNotFoundError: If the STL file doesn't exist
    """
    mesh = trimesh.load_mesh(stl_file_name)
    edges = mesh.edges_unique_length
    average_edge_length = np.mean(edges)
    return average_edge_length

def tetrahedral_mesher(stl_file_name, detail_level=None):
    """
    Generate a tetrahedral volume mesh from an STL surface mesh.

    Args:
        stl_file_name (str): Path to the input STL file
        detail_level (float, optional): Desired mesh resolution. If None, automatically estimated

    Returns:
        pyvista.UnstructuredGrid: Volumetric tetrahedral mesh
        None: If mesh generation fails

    Raises:
        FileNotFoundError: If the STL file doesn't exist
        ValueError: If tetrahedral cells are not generated
    """
    if not os.path.exists(stl_file_name):
        raise FileNotFoundError(f"STL file '{stl_file_name}' does not exist.")

    ascii_stl_file = convert_stl_to_ascii(stl_file_name, stl_file_name)

    if detail_level is None:
        detail_level = estimate_surface_element_size(stl_file_name)
        print(f"Estimated surface element size: {detail_level}")

    if not gmsh.isInitialized():
        gmsh.initialize()
    else:
        print("Gmsh has already been initialized, proceeding without reinitialization.")

    try:
        with pygmsh.geo.Geometry() as geom:
            # Configure Gmsh options
            gmsh.option.setNumber("General.Terminal", 1)
            characteristic_length = detail_level
            
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", characteristic_length)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", characteristic_length)
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.QualityType", 2)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

            try:
                gmsh.merge(ascii_stl_file)
            except Exception as e:
                raise Exception(f"Failed to load ASCII STL file: {ascii_stl_file}. Error: {e}")

            # Create volume from surface
            dimension = gmsh.model.getDimension()
            entities = gmsh.model.getEntities(dimension)
            surface_loop = gmsh.model.geo.addSurfaceLoop([entity[1] for entity in entities])
            gmsh.model.geo.addVolume([surface_loop])

            gmsh.model.geo.synchronize()
            mesh = geom.generate_mesh(dim=3, verbose=False, algorithm=8)

            if 'tetra' not in mesh.cells_dict:
                raise ValueError("Tetrahedral cells were not generated.")

            # Convert to PyVista format
            tetra_cells = mesh.cells_dict['tetra']
            points = mesh.points
            cells = np.hstack([[4] + list(cell) for cell in tetra_cells])
            points = np.array(points)
            cell_types = np.full(len(tetra_cells), pv.CellType.TETRA, dtype=np.uint8)

            return pv.UnstructuredGrid(cells.astype(np.int32), cell_types, points)

    except Exception as e:
        print(f"An error occurred during mesh generation: {e}")
        return None
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()

def load_ct_image(file_path):
    """
    Load a CT image from file (supports DICOM, NIFTI, etc.).

    Args:
        file_path (str): Path to the CT image file

    Returns:
        SimpleITK.Image: Loaded CT image

    Raises:
        RuntimeError: If the image cannot be loaded
    """
    return sitk.ReadImage(file_path)

def get_labels(ct_image):
    """
    Extract unique label values from a labeled CT image.

    Args:
        ct_image (SimpleITK.Image): Labeled CT image

    Returns:
        list: List of unique label values (excluding background label 0)
    """
    label_stats = sitk.LabelStatisticsImageFilter()
    label_stats.Execute(ct_image, ct_image)
    labels = list(label_stats.GetLabels())
    if 0 in labels:
        labels.remove(0)
    return labels

def apply_morphological_operation(label_mask, operation='dilate', radius=1):
    """
    Apply morphological operations to a binary mask.

    Args:
        label_mask (SimpleITK.Image): Binary mask
        operation (str): Type of operation ('dilate', 'erode', 'opening', 'closing')
        radius (int): Radius of the structural element

    Returns:
        SimpleITK.Image: Processed binary mask
    """
    radius_vector = [radius] * label_mask.GetDimension()
    
    operations = {
        'dilate': sitk.BinaryDilate,
        'erode': sitk.BinaryErode,
        'opening': sitk.BinaryMorphologicalOpening,
        'closing': sitk.BinaryMorphologicalClosing
    }
    
    if operation in operations:
        return operations[operation](label_mask, radius_vector)
    return label_mask

def check_and_clean_mesh(pv_mesh):
    """
    Clean and validate a PyVista mesh.

    Args:
        pv_mesh (pyvista.PolyData): Input mesh

    Returns:
        pyvista.PolyData: Cleaned mesh
    """
    cleaned_mesh = pv_mesh.clean(tolerance=1e-6)
    if cleaned_mesh.n_points < pv_mesh.n_points:
        print(f"Mesh cleaned: {pv_mesh.n_points} -> {cleaned_mesh.n_points} points")
    return cleaned_mesh

def surface_mesher(ct_image, label_value, spacing, origin, operation=None, radius=1, 
                  n_iter=30, relaxation_factor=0.1, subdivide=3, element_size=None):
    """
    Generate a surface mesh from a labeled CT image.

    Args:
        ct_image (SimpleITK.Image): Input CT image
        label_value (int): Label value to mesh
        spacing (tuple): Voxel spacing (x, y, z)
        origin (tuple): Image origin coordinates
        operation (str, optional): Morphological operation type
        radius (int): Radius for morphological operations
        n_iter (int): Number of smoothing iterations
        relaxation_factor (float): Smoothing relaxation factor
        subdivide (int): ACVD subdivision level
        element_size (float, optional): Target element size

    Returns:
        pyvista.PolyData: Surface mesh
    """
    # Create binary mask
    label_mask = sitk.BinaryThreshold(ct_image, lowerThreshold=label_value, 
                                    upperThreshold=label_value)

    if operation is not None:
        label_mask = apply_morphological_operation(label_mask, operation, radius)
    
    label_mask_np = sitk.GetArrayFromImage(label_mask)
    label_mask_np = np.swapaxes(label_mask_np, 0, 2)

    # Generate surface using marching cubes
    verts, faces, normals, _ = measure.marching_cubes(label_mask_np, level=0, spacing=spacing)
    verts += np.array(origin)

    # Create PyVista mesh
    faces_fixed = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32).ravel()
    pv_mesh = pv.PolyData(verts, faces_fixed)

    # Apply Laplacian smoothing
    if n_iter > 0:
        pv_mesh = pv_mesh.smooth(n_iter=n_iter, relaxation_factor=relaxation_factor)

    # Apply ACVD remeshing if element size is specified
    if element_size is not None:
        total_area = pv_mesh.area
        target_points = int(total_area / element_size / 2)
        print(f"Target points for label {label_value}: {target_points}")
        
        clust = pyacvd.Clustering(pv_mesh)
        clust.subdivide(subdivide)
        clust.cluster(target_points)
        pv_mesh = clust.create_mesh()

    return check_and_clean_mesh(pv_mesh)

def save_mesh(pv_mesh, output_filename):
    """
    Save a PyVista mesh to file.

    Args:
        pv_mesh (pyvista.PolyData): Mesh to save
        output_filename (str): Path to save the mesh
    """
    pv_mesh.save(output_filename)
    print(f"Surface saved to: {output_filename}")

def process_single_label(label_value, ct_image, spacing, origin, output_dir, 
                        file_format, operation, radius, n_iter, 
                        relaxation_factor, subdivide, element_size):
    """
    Process a single label from a CT image and generate a surface mesh.

    Args:
        label_value (int): Label to process
        ct_image (SimpleITK.Image): CT image
        spacing (tuple): Voxel spacing
        origin (tuple): Image origin
        output_dir (str): Output directory
        file_format (str): Output file format ('stl' or 'obj')
        operation (str): Morphological operation type
        radius (int): Morphological operation radius
        n_iter (int): Smoothing iterations
        relaxation_factor (float): Smoothing factor
        subdivide (int): ACVD subdivision level
        element_size (float): Target element size
    """
    mesh = surface_mesher(ct_image, label_value, spacing, origin, operation, 
                         radius, n_iter, relaxation_factor, subdivide, element_size)
    output_filename = f"{output_dir}/label_{label_value}.{file_format}"
    save_mesh(mesh, output_filename)

def process_ct_labels(ct_image, output_dir, file_format="stl", operation=None, 
                     radius=1, n_iter=30, relaxation_factor=0.1, 
                     subdivide=3, element_size=None):
    """
    Process all labels in a CT image and generate surface meshes.

    Args:
        ct_image (SimpleITK.Image): Input CT image
        output_dir (str): Output directory for meshes
        file_format (str): Output file format ('stl' or 'obj')
        operation (str, optional): Morphological operation type
        radius (int): Radius for morphological operations
        n_iter (int): Number of smoothing iterations
        relaxation_factor (float): Smoothing factor
        subdivide (int): ACVD subdivision level
        element_size (float, optional): Target element size
    """
    spacing = ct_image.GetSpacing()
    origin = ct_image.GetOrigin()
    print(f"Voxel spacing: {spacing}")
    print(f"Voxel origin: {origin}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")

    labels = get_labels(ct_image)
    labels.sort()
    print(f"Found labels (excluding background): {labels}")

    for label in labels:
        process_single_label(label, ct_image, spacing, origin, output_dir, 
                           file_format, operation, radius, n_iter, 
                           relaxation_factor, subdivide, element_size)

def assemble_pelvis(bone_list_path, file_to_save='pelvic.vtk'):
    """
    Assemble multiple bone meshes into a single pelvic model.

    Args:
        bone_list_path (list): List of paths to bone mesh files
        file_to_save (str): Output file path for the assembled model

    Returns:
        pyvista.UnstructuredGrid: Assembled pelvic mesh
    """
    for i, bone_file in enumerate(bone_list_path):
        mesh = tetrahedral_mesher(bone_file, None)
        if i == 0:
            pelvis = mesh.copy()
        else:
            pelvis = pelvis.merge(mesh)
    pelvis.save(file_to_save)
    return pelvis

if __name__ == '__main__':
    # Example usage
    file_path = "templates/segmentation/template_labels.nrrd"
    output_dir = "templates/geometry/base"
    file_format = "stl"
    operation = 'erode'
    radius = 1
    n_iter = 30
    relaxation_factor = 0.1
    subdivide = 3
    element_size = 2.0

    ct_image = load_ct_image(file_path)
    process_ct_labels(ct_image, output_dir, file_format, operation, radius, 
                     n_iter, relaxation_factor, subdivide, element_size)
    
    # Assemble pelvic model
    stl_filename_prefix = 'templates/geometry/base/'
    bone_list = [
        stl_filename_prefix + f'label_{i}.stl' for i in range(1, 6)
    ]
    pelvis_file_name = stl_filename_prefix + 'pelvic.vtk'
    assemble_pelvis(bone_list, pelvis_file_name)