# Data for: Statistical Study of Atomic, Electronic, and Mechanical Properties of Amorphous Graphene

This repository contains the dataset and analysis scripts for our study on amorphous graphene (a-G), published in *Physical Review B*.

## 1. Citation
If you use these materials, please cite:
> Jun Chai, et al. "Statistical Study of Atomic, Electronic, and Mechanical Properties of Amorphous Graphene." Phys. Rev. B (2026). [under review]

## 2. Directory Structure
- `data/`: Contains `100_structures.rar`, which includes 100 statistically representative models of a-G (in POSCAR/XYZ format).
- `src/`: 
    - `ring_v7.py`: Python script used for ring-size distribution analysis (identifying 5, 6, 7, 8-membered rings).
- `LICENSE`: MIT License.

## 3. Data Description
The models were generated using MLFF-accelerated AIMD via the helium potential-wall method. 
- **Atomic Structures:** 100 independent samples to ensure statistical convergence.
- **Ring Analysis:** The `ring_v7.py` script identifies the Continuous Random Network (CRN) topology.

## 4. Usage
To extract the structures:
```bash
unrar x data/100_structures.rar
