import ants
from tqdm import tqdm
from utils import shift_intensity, create_dir
import database as db
from pathlib import Path

def apply_rigid_registration(mi, fi, files_path):
    """
    Applies rigid registration to align a moving image with a fixed (template) image.
    
    This function performs the following steps:
    1. Normalizes the intensity of the moving image
    2. Performs rigid registration using ANTs
    3. Applies the transformation to the original image
    4. Saves the registered image
    
    Args:
        mi (ants.ANTsImage): Moving image to be registered
        fi (ants.ANTsImage): Fixed (template) image to register against
        files_path (Path): Directory path where output files will be saved
        
    Returns:
        None
        
    Outputs:
        - Saves the rigidly registered image as 'rigid.nii.gz' in the specified directory
        
    Note:
        The function preserves the original intensity range of the moving image by
        applying intensity shifts before and after registration.
    """
    int_offset = mi.numpy().min()
    mi_orig = shift_intensity(mi, -int_offset)
    mi = ants.iMath_normalize(mi_orig)

    # Perform rigid registration
    mytx = ants.registration(fixed=fi, moving=mi, 
                           type_of_transform='DenseRigid', 
                           verbose=False)
    
    # Apply transforms to original image
    wi = ants.apply_transforms(fixed=fi, moving=mi_orig, 
                             transformlist=mytx['fwdtransforms'])

    # Shift intensity back to original offset
    ants.image_write(shift_intensity(wi, +int_offset), 
                    str(files_path / 'rigid.nii.gz'))
    
def apply_syn_registration(mi, fi, files_path):
    """
    Applies SyN (Symmetric Normalization) deformable registration to align images.
    
    This function performs advanced deformable registration using ANTs' SyNAggro
    transform, which includes both affine and deformable components. It saves both
    the warped image and the transformation files for forward and inverse mappings.
    
    Args:
        mi (ants.ANTsImage): Moving image to be registered (typically the rigid-aligned image)
        fi (ants.ANTsImage): Fixed (template) image to register against
        files_path (Path): Directory path where output files will be saved
        
    Returns:
        None
        
    Outputs:
        Several files are saved in the specified directory:
        - warped.nii.gz: The final registered image
        - afn_fwd.mat: Forward affine transformation matrix
        - warp_fwd.nii.gz: Forward warp field
        - afn_inv.mat: Inverse affine transformation matrix
        - warp_inv.nii.gz: Inverse warp field
        
    Registration Parameters:
        - Metric: Mattes Mutual Information for both affine and SyN stages
        - Sampling: 32 points
        - Affine iterations: [2100, 1200, 1200, 100]
        - Deformable iterations: [100, 80, 50, 30]
        - Random sampling rate: 0.2
    """
    int_offset = mi.numpy().min()
    mi_orig = shift_intensity(mi, -int_offset)
    mi = ants.iMath_normalize(mi_orig)
        
    mytx = ants.registration(fi, mi, 
                           aff_metric='mattes', 
                           syn_metric='mattes',
                           syn_sampling=32, 
                           aff_sampling=32,
                           aff_iterations=[2100, 1200, 1200, 100],
                           reg_iterations=[100, 80, 50, 30],
                           type_of_transform='SyNAggro',
                           aff_random_sampling_rate=0.2)

    wi = ants.apply_transforms(fixed=fi, 
                             moving=mi_orig,
                             transformlist=mytx['fwdtransforms'])
    
    wi = shift_intensity(wi, + int_offset)
    
    # Save warped image
    ants.image_write(wi, str(files_path / 'warped.nii.gz'))

    # Save forward transformations
    afftxfwd = ants.read_transform(mytx['fwdtransforms'][1])
    warptxfwd = ants.image_read(mytx['fwdtransforms'][0])
    ants.write_transform(afftxfwd, str(files_path / 'afn_fwd.mat'))
    ants.image_write(warptxfwd, str(files_path / 'warp_fwd.nii.gz'))

    # Save inverse transformations
    afftxinv = ants.read_transform(mytx['invtransforms'][0])
    warptxinv = ants.image_read(mytx['invtransforms'][1])
    ants.write_transform(afftxinv, str(files_path / 'afn_inv.mat'))
    ants.image_write(warptxinv, str(files_path / 'warp_inv.nii.gz'))

if __name__ == '__main__':

    # Define paths to database directories
    database_directory = Path('/mnt/database/BoneDat')
    database_raw = database_directory / 'raw'
    database_registration = database_directory / 'derived' / 'registrations'
    
    # Collect patient information from database
    patient_info = db.collect_patient_info(database_raw)

    # Load and normalize template image
    template = ants.image_read('templates/segmentation/template.nii.gz')
    template = ants.iMath_normalize(template)

    # Process each patient's images
    for patient_folder in tqdm(patient_info, desc='registering images...'):
        patient_path = database_raw / patient_folder
        mi = ants.image_read(str(patient_path/ 'original.nii.gz'))

        reg_path = database_registration / patient_folder
        create_dir(reg_path)

        # Apply registration pipeline: rigid followed by SyN
        apply_rigid_registration(mi, template, reg_path)
        mi = ants.image_read(str(reg_path / 'rigid.nii.gz'))
        apply_syn_registration(mi, template, reg_path)