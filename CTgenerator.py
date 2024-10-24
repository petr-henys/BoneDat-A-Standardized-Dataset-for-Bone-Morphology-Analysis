import pyvista as pv
import numpy as np
from pathlib import Path
import ants
from utils import create_dir
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import openturns as ot

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class PCAsampler:
    """
    This class performs Principal Component Analysis (PCA) and samples new data 
    from the learned PCA space using Latin Hypercube Sampling (LHS).

    Args:
        x (np.ndarray): Input data matrix with shape (n_samples, n_features).
        n_components (int): Number of principal components to keep.

    Attributes:
        x_orig (np.ndarray): Original input data.
        pca (sklearn.decomposition.PCA): Fitted PCA object.
        stdscaler (sklearn.preprocessing.StandardScaler): Fitted StandardScaler object.
        LHSSIMS (int): Number of simulations for Latin Hypercube Sampling.
        distro (openturns.ComposedDistribution): Distribution for LHS, a normal distribution 
                                                for each principal component.

    Methods:
        sample(ns): Generates new samples from the PCA space.
    """
    def __init__(self, x, n_components):
        """
        Initializes the PCAsampler with data and number of components.

        Performs standardization, fits PCA on the standardized data, and 
        sets up the distribution for Latin Hypercube Sampling.
        """
        self.x_orig = x
        self.pca = PCA(n_components, whiten=True)

        self.stdscaler = StandardScaler(with_mean=True, with_std=True)

        Xs = self.stdscaler.fit_transform(x)
        self.pca.fit(Xs)

        self.LHSSIMS = 100 
        distros = n_components * [ot.Normal(0.0, 1.0)]
        self.distro = ot.ComposedDistribution(distros, ot.IndependentCopula(len(distros)))

    def sample(self, ns):
        """
        Generates new samples from the PCA space.

        Args:
            ns (int): Number of samples to generate.

        Returns:
            np.ndarray: Generated samples in the original data space.
        """
        lhs = ot.LHSExperiment(self.distro, ns)
        lhs.setAlwaysShuffle(True)
        # Strategy to ensure uncorrelated structure in data
        spaceFilling = ot.SpaceFillingC2()
        lhs = ot.MonteCarloLHS(lhs, self.LHSSIMS, spaceFilling)
        z = np.asarray(lhs.generate())
        new_X = self.pca.inverse_transform(z)
        return self.stdscaler.inverse_transform(new_X)

class CombineSplitArrays:
    """
    This class combines and splits arrays X and I with potentially different 
    sizes in the M dimension.

    Attributes:
        M_X (int): Size of the M dimension for array X.
        M_I (int): Size of the M dimension for array I.
        dim (int): Size of the geometric dimension.

    Methods:
        combine_arrays(X, I): Combines arrays X and I into a single array.
        split_array(combined): Splits the combined array back into X and I.
    """
    def __init__(self):
        """
        Initializes the CombineSplitArrays class.
        """
        self.M_X = None  # Size of the M dimension for array X
        self.M_I = None  # Size of the M dimension for array I
        self.dim = None  # Size of the geometric dimension

    def combine_arrays(self, X, I):
        """
        Combines X[N, M_X, dim] and I[N, M_I] into a single array 
        [N, M_X * dim + M_I].

        Args:
            X (np.ndarray): Array with shape (N, M_X, dim).
            I (np.ndarray): Array with shape (N, M_I).

        Returns:
            np.ndarray: Combined array.
        """
        N, M_X, dim = X.shape
        N_I, M_I = I.shape
        assert N == N_I, "Arrays X and I must have the same number of rows (N)."

        # Store M_X, M_I, and dim for later reconstruction
        self.M_X = M_X
        self.M_I = M_I
        self.dim = dim

        # Reshape X to a 2D array [N, M_X * dim]
        X_reshaped = X.reshape(N, -1)

        # Combine X and I along the last axis
        combined = np.concatenate([X_reshaped, I], axis=-1)

        return combined

    def split_array(self, combined):
        """
        Splits the combined array back into its original structure X[N, M_X, dim] 
        and I[N, M_I].

        Args:
            combined (np.ndarray): Combined array.

        Returns:
            tuple: A tuple containing the reconstructed X and I arrays.
        """
        N = combined.shape[0]

        assert self.M_X is not None and self.M_I is not None and self.dim is not None, \
              "Arrays were not combined beforehand."

        # Calculate the split point
        split_point = self.M_X * self.dim

        # Extract X and I
        X_reconstructed = combined[:, :split_point].reshape(N, self.M_X, self.dim)
        I_reconstructed = combined[:, split_point:]

        return X_reconstructed, I_reconstructed

def process_displacement(mesh, Ui, image):
    """
    Generates a displacement field by fitting a B-spline to the given displacements.

    Args:
        mesh (pyvista.PolyData): The mesh containing the displacement origins.
        Ui (numpy.ndarray): The displacements to fit.
        image (ants.ANTsImage): The image to use as a reference for the displacement field.

    Returns:
        ants.ANTsTransform: The displacement field transform.
    """
    displacement = ants.fit_bspline_displacement_field(
        displacement_origins=mesh.points,
        displacements=Ui,
        origin=image.origin,
        spacing=image.spacing,
        size=image.shape,
        direction=image.direction,
        number_of_fitting_levels=7,
        estimate_inverse=False,
        mesh_size=2
    )
    return ants.transform_from_displacement_field(displacement)

def apply_transformation(mytx, mesh, image):
    """
    Applies a transformation to an image and a mesh.

    Args:
        mytx (ants.ANTsTransform): The transformation to apply.
        mesh (pyvista.PolyData): The mesh to transform.
        image (ants.ANTsImage): The image to transform.

    Returns:
        tuple: A tuple containing the warped image and the transformed mesh points.
    """
    warped_img = ants.apply_ants_transform_to_image(mytx, image, image)
    X_est = np.array([mytx.apply_to_point(point) for point in mesh.points])
    return (warped_img, X_est)

def save_synthetic_patient(patient_index, warped_img, X_est, U_ref, synthetic_patients_folder):
    """
    Saves the data for a synthetic patient.

    Args:
        patient_index (int): The index of the patient.
        warped_img (ants.ANTsImage): The warped image.
        X_est (numpy.ndarray): The estimated deformation.
        U_ref (numpy.ndarray): The reference deformation.
        synthetic_patients_folder (str): The folder to save the patient data to.
    """
    patient_folder = Path(synthetic_patients_folder) / f'patient_{patient_index}'
    create_dir(patient_folder)
    ants.image_write(warped_img, str(patient_folder / 'original.nii.gz'))
    np.save(patient_folder / 'est_deformation.npy', X_est)
    np.save(patient_folder / 'ref_deformation.npy', U_ref)

if __name__ == '__main__':
    synthetic_patients_folder = '/mnt/database/synthetic_patients/raw'
    # Load datasets
    with logging_redirect_tqdm():
        logger.info("Loading X and I data...")
        mesh = pv.read('template/pelvic.vtk')
        X = np.load('additional_data/X_inv.npy')
        I = np.load('additional_data/I_voxels.npy')
        U = X - mesh.points[None, :, :]
        
        splitter = CombineSplitArrays()
        XI = splitter.combine_arrays(U, I)
        
        # PCA sampling
        logger.info("Performing PCA sampling...")
        sampler = PCAsampler(XI, len(X)-1)
        new_XI = sampler.sample(len(X))
        new_U, new_I = splitter.split_array(new_XI)

        create_dir(synthetic_patients_folder)

        # Load template images and meshes
        logger.info("Loading templates...")
        template = ants.image_read('template/template.nii.gz')
        mask = template > 1e-2 # remove background
        
        # Process each synthetic patient with progress bar
        logger.info("Processing synthetic patients...")
        for i, Ui in tqdm(enumerate(new_U), total=len(new_U), desc="Processing Patients", ncols=100):
            # Fit BSpline to generate displacement field
            template[mask] = new_I[i]
            mytx = process_displacement(mesh, Ui, template)

            # Apply transformation and save results
            warped_img, X_est = apply_transformation(mytx, mesh, template)
            save_synthetic_patient(i, warped_img, X_est, Ui, synthetic_patients_folder)
            logger.info(f"Patient {i} processed and saved.")