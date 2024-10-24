Sure, here is a well-structured `readme.md` for the repository containing your Python scripts, incorporating best practices for clarity and reproducibility:

```
# BoneDat: A Comprehensive Database of Standardized Bone Morphology for In-silico Analyses

## Overview
This repository contains Python scripts and documentation for the BoneDat project, a comprehensive database of standardized bone morphology data derived from 250 clinical lumbopelvic CT scans. The scripts cover various aspects of the project, including:

- Template building
- Image registration (rigid and deformable)
- Registration quality assessment
- Mesh generation and analysis
- Anthropometric analysis

This README provides information on setting up the environment, running the scripts, and understanding the project workflow.

## Project Structure

├── MI_metric_evaluation.py
├── generate_patient_meshes.py
├── mesh_quality.py
├── antropometric_analysis.py
├── collect_intensity_in_voxels.py
├── build_template.py
├── registration.py
├── additional_data
│   ├── ref_angles.xlsx
│   ├── I_voxels.npy
│   ├── mesh_quality.vtk
│   ├── ref_distances.xlsx
│   ├── registration_metrics.xlsx
│   ├── points4distances.npy
│   └── points4angles.npy
├── templates
│   ├── geometry
│   │   └── base
│   │       └── pelvic.vtk
│   └── new_template.mha
└── database
    └── BoneDat
        ├── raw
        └── derived
            ├── registrations
            └── fields

This tree-like structure is much clearer and more informative, giving a better visual representation of how the files and directories are organized within the project.
```

## Dependencies

- Python 3.11+ (recommended)
- ANTs (Advanced Normalization Tools)
- pandas
- numpy
- pathlib
- scipy
- tqdm
- pyvista
- meshio
- pygmsh

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/BoneDat.git](https://github.com/your-username/BoneDat.git)
   ```

2. Create and activate a virtual environment on Linux (Ubuntu 24.04 LTS recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install ANTs:
   ```bash
   # Follow instructions specific to your operating system
   # [https://antspy.readthedocs.io/en/latest/installation.html](https://antspy.readthedocs.io/en/latest/installation.html)
   ```

## Usage

### Template Building

1. Ensure the `database` directory is correctly configured with your data.
2. Run the `build_template.py` script:
   ```bash
   python build_template.py
   ```

### Image Registration

1. Ensure the `database` directory is correctly configured with your data.
2. Run the `registration.py` script:
   ```bash
   python registration.py
   ```

### Mesh Generation

1. Ensure the `database` directory is correctly configured with your data.
2. Run the `generate_patient_meshes.py` script:
   ```bash
   python generate_patient_meshes.py
   ```

### Mesh Quality Analysis

1. Ensure the `database` directory is correctly configured with your data.
2. Run the `mesh_quality.py` script:
   ```bash
   python mesh_quality.py
   ```

### Anthropometric Analysis

1. Ensure the `database` directory is correctly configured with your data.
2. Run the `anthropometric_analysis.py` script:
   ```bash
   python antropometric_analysis.py
   ```

## Documentation

- Detailed documentation for each script is provided in the `docs` directory.
- Refer to the manuscript for a comprehensive description of the project and its findings.

## Contributing

Contributions to the project are welcome. Please follow the standard GitHub workflow for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This research was supported by the Czech Science Foundation Grant No. 24-10862S.

## Contact

For any questions or inquiries, please contact Michal Kuchař at michal.kuchar@tul.cz or Petr Henyš at petr.henys@tul.cz.
```
