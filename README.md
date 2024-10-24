
This repository contains  accompanying code used to reproduce the results of **BoneDat** article, a comprehensive database for standardized bone morphology extracted from 278 clinical lumbopelvic CT scans. This work focuses on enhancing the reproducibility of bone morphology research, critical for biomechanics, pathology, and clinical studies.

## Article Overview

**BoneDat: A Database of Standardized Bone Morphology for In-Silico Analyses** presents the development of a bone morphology database derived from pelvic and lower spine CT scans. The database includes age and sex-stratified lumbopelvic scans that were normalized and segmented for detailed in-silico analysis.

### Key Contributions:
- **BoneDat Database**: A dataset of standardized pelvic and spinal bone CT scans for research in bone biomechanics and pathology.
- **Shape Normalization**: Provides a method for shape normalization using the Advanced Normalization Tools (ANTs) for geometric alignment across patients.
- **Volume Mesh Creation**: Tetrahedral meshes were generated for computational simulations based on the morphed bone models.

## Software Overview

### Mesh Generation Pipeline

The script in this repository supports the generation of volumetric meshes from segmented bone data, facilitating computational modeling. Key stages include:
- **Segmentation**: Bone morphology is extracted using Biomedisa’s sparse interpolation.
- **Template Generation**: A statistical model of bone shape is generated using ANTs and registered across multiple samples.
- **Mesh Morphing**: The template mesh is morphed to match individual samples, allowing for the generation of subject-specific computational models.

### Registration and Shape Normalization

Bone shape normalization ensures that all samples are aligned to a reference shape, which is critical for comparative morphology studies. Using the ANTs library, the following transformation steps are applied:
- **Mass Center Alignment**
- **Rigid Transformation**
- **Affine Transformation**
- **Non-linear SyN Transformation**

These transformations minimize deformations between the dataset bones and the template, allowing for accurate population-level comparisons.

### Volume Mesh Quality

The tetrahedral meshes generated in this project are of high quality, with an aspect ratio up to 2.5 ± 1.5, suitable for in-silico studies. Post-processing with **Gmsh** is performed to improve mesh quality and ensure stability in computational models.

### Validation

The process was validated through:
- **Shape Normalization Robustness**: Quantitative assessments show minimal error in aligning bone shapes across individuals, with mutual information metrics used to assess registration quality.
- **Mesh Quality**: The aspect ratio distribution of tetrahedral elements ensures numerical stability and accuracy in simulations.

### Tools and Libraries
- **Python**: Used for data manipulation and processing (SimpleITK, ANTsPy, PyVista).
- **ANTs**: Advanced Normalization Tools for medical image registration and shape normalization.
- **Gmsh**: For mesh post-processing and quality improvement.
- **Paraview/MITK**: For data visualization.

## Repository Structure

```plaintext
- manuscript.pdf              # Scientific article
- mesh-generation-pipeline.md # Documentation for the mesh generation process
- registration-error-analysis.md # Analysis of registration errors and validation
- registration-pipeline-docs.md # Detailed documentation on the shape normalization and registration pipeline
- mesher.py                   # Python script for mesh generation and processing
```

### Instructions for Reproducing Results

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/bone-dat.git
   cd bone-dat
   ```

2. **Set up Python environment**:
   The required Python packages are listed in the `requirements.txt`. Create and activate a virtual environment:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the Mesh Generation Pipeline**:
   Use the provided `mesher.py` script to generate tetrahedral meshes for the dataset:
   ```bash
   python mesher.py
   ```

4. **View the Generated Meshes**:
   The meshes can be visualized using Paraview or MITK. Detailed instructions are provided in the `registration-pipeline-docs.md` file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, please reach out to:
- Michal Kuchar (michal.kuchar@tul.cz)

--- 

This README provides a comprehensive overview of the article, code, and instructions for reproducing the in-silico bone morphology analyses.
