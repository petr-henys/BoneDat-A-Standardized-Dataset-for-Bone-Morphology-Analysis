# BoneDat: A Comprehensive Database of Standardized Bone Morphology for In-silico Analyses

## Overview
This repository contains Python scripts and documentation for the [BoneDat project](https://zenodo.org/uploads/13970522), a comprehensive database of standardized bone morphology data derived from 278 clinical lumbopelvic CT scans. The scripts cover various aspects of the project, including:

- Template building
- Image registration (rigid and deformable)
- Registration quality assessment
- Mesh generation and quality assessment
- Anthropometric analysis

This README provides information on setting up the environment, running the scripts, and understanding the project workflow.
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
