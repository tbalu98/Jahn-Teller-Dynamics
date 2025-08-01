# Jahn-Teller-Dynamics - Dynamic Jahn-Teller Effect Calculator


[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Email](https://img.shields.io/badge/Email-toth.balazs%40wigner.hun--ren.hu-blue)](mailto:toth.balazs@wigner.hun-ren.hu)



A Python tool for calculating dynamic Jahn-Teller effects, spin-orbit interactions, and energy splitting of degenerate electron states from DFT calculations.

## Table of Contents

- [About the Project](#about-the-project)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration Examples](#configuration-examples)
- [Output Options](#output-options)
- [Eigenstates](#eigenstates)

## About the Project

The `Exe.py` script calculates:
* The E⊗e dynamic Jahn-Teller effect
* Spin-orbit interaction
* Energy splitting of degenerate electron states
* ZPL (zero phonon line) shift in magnetic field

from results of DFT calculations.

This project was built with Python 3.10.

## Installation

### Basic Installation

1. Download the project:
   ```bash
   git clone https://github.com/tbalu98/Jahn-Teller-Dynamics.git
   ```
   or download the [zip file](https://github.com/tbalu98/Jahn-Teller-Dynamics/archive/refs/heads/main.zip)

2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

### Development Setup

1. Clone the repository:
   ```bash
      git clone https://github.com/tbalu98/Jahn-Teller-Dynamics.git
   cd DJT
   ```

2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

3. Set up your development environment:
   ```bash
   git remote set-url origin https://github.com/your-username/Jahn-Teller-Dynamics.git
   git remote -v  # Verify the changes
   ```

## Directory Structure




## Usage

The script requires a configuration file to specify your Jahn-Teller active system:

```bash
python3 Exe.py configuration_file.cfg
```

### Test Data

To test the code, download this folder that contains the vasprunxml files for the tin vacancy (SnV) from:
[Google Drive](https://drive.google.com/drive/folders/1_L-j08JW8fuGhRihFijjS_vfALAPvC43?usp=sharing)

## Configuration Examples

Example configuration files can be found in the `config_files` folder, which demonstrate different use cases and parameter settings.

### 1. Basic Configuration with vasprunxml

To perform calculations, you need at least two geometric configurations of your Jahn-Teller active system:
- The highly symmetric geometry
- The Jahn-Teller distorted geometry with minimum energy
- (Optional) The saddle point geometry

Example configuration:
```
[essentials]
maximum_number_of_vibrational_quanta = 12
output_prefix = SnV_gnd
output_folder = results/SnV_results
input_folder = data/SnV_data
eigen_states = real
save_raw_parameters = true
save_model_hamiltonian_cfg = true


[system_parameters]
saddle_point_geometry = SnV_gnd_saddle_point.xml
global_minimum_geometry = SnV_gnd_C2h.xml
high_symmetry_geometry = SnV_gnd_D3d.xml
#spin-orbit coupling obtained from DFT calculation in meV:
#This calculation modells a hole therefore, it is a negative value.
DFT_spin-orbit_coupling = -8.3
orbital_reduction_factor =  0.328
```

The eigenenergies, states and the theoretical results such as reduction factors will be presented in the output_folder.

### 2. ZPL Calculation Configuration

For ZPL shift calculations, you need to specify:
- Ground and excited states
- Magnetic field parameters
- Basis vectors of the Jahn-Teller active system

```
[essentials]
maximum_number_of_vibrational_quanta = 12
output_prefix = SnV_ZPL_shift
output_folder = results/SnV_results
input_folder = data/SnV_data
save_raw_parameters = true
#basis vectors of the Exe system's coordinate system
basis_vector_1 = 1, -1, 0
basis_vector_2 = 1, 1, -2
basis_vector_3 = 1, 1, 1
model_Hamiltonian = true
save_model_Hamiltonian_cfg = true

[ground_state_parameters]
saddle_point_geometry = SnV_gnd_saddle_point.xml
global_minimum_geometry = SnV_gnd_C2h.xml
high_symmetry_geometry = SnV_gnd_D3d.xml
#spin-orbit coupling obtained from DFT calculation in meV:
DFT_spin-orbit_coupling = -8.3
orbital_reduction_factor =  0.328


[excited_state_parameters]
saddle_point_geometry = SnV_ex_saddle_point.xml
global_minimum_geometry = SnV_ex_C2h.xml
high_symmetry_geometry = SnV_ex_D3d.xml
#spin-orbit coupling obtained from DFT calculation in meV:
DFT_spin-orbit_coupling = -95.9
orbital_reduction_factor =  0.782


[magnetic_field]
#In Tesla
minimum = 0.0
maximum = 10.0
direction_vector = 1.0, 0.0, 0.0
step_number = 11
```

### 3. CSV File Configuration

When using CSV files for atomic coordinates, the configuration includes additional structural parameters:

```
[essentials]
maximum_number_of_vibrational_quanta = 12
output_prefix = SnV_ZPL_shift
output_folder = results/SnV_results
input_folder = data/SnV_data
save_raw_parameters = false
basis_vector_1 = 1, -1, 0
basis_vector_2 = 1, 1, -2
basis_vector_3 = 1, 1, 1
model_hamiltonian = true
save_model_hamiltonian_cfg = true


[ground_state_parameters]
dft_spin-orbit_coupling = -8.3
orbital_reduction_factor = 0.328
high_symmetry_geometry = SnV_ZPL_shift_ground_state_parameters_high_symmetry_geometry.csv
high_symmetric_geometry_energy = -5368.30679265
global_minimum_geometry = SnV_ZPL_shift_ground_state_parameters_global_minimum_geometry.csv
global_minimum_energy = -5368.32839163
saddle_point_energy = -5368.32682965
saddle_point_geometry = SnV_ZPL_shift_ground_state_parameters_saddle_point_geometry.csv

[excited_state_parameters]
dft_spin-orbit_coupling = -95.9
orbital_reduction_factor = 0.782
high_symmetry_geometry = SnV_ZPL_shift_excited_state_parameters_high_symmetry_geometry.csv
high_symmetric_geometry_energy = -5366.13656085
global_minimum_geometry = SnV_ZPL_shift_excited_state_parameters_global_minimum_geometry.csv
global_minimum_energy = -5366.21970743
saddle_point_energy = -5366.21291581
saddle_point_geometry = SnV_ZPL_shift_excited_state_parameters_saddle_point_geometry.csv

[atom_structure_parameters]
masses_of_atoms = 118.71, 12.011
names_of_atoms = Sn, C
numbers_of_atoms = 1, 510
basis_vector_1 = 14.17860889, 0.0, 0.0
basis_vector_2 = 0.0, 14.17860889, 0.0
basis_vector_3 = 0.0, 0.0, 14.17860889



[magnetic_field]
minimum = 0.0
maximum = 10.0
step_number = 11
direction_vector = 1.0, 0.0, 0.0
```

### 4. Direct Parameter Configuration

You can also specify Jahn-Teller interaction parameters directly:

```
[essentials]
maximum_number_of_vibrational_quanta = 12
output_prefix = SnV_ZPL_shift_JT_pars
output_folder = results/SnV_results
spectrum_range = 50
#model_Hamiltonian = true
#basis vectors of the Exe system's coordinate system
basis_vector_1 = 1, -1, 0
basis_vector_2 = 1, 1, -2
basis_vector_3 = 1, 1, 1
#save_model_Hamiltonian_cfg = true


[ground_state_parameters]
Jahn-Teller_energy = 21.599
barrier_energy = 1.562
vibrational_energy_quantum = 79.4954
#spin-orbit coupling obtained from DFT calculation in meV:
DFT_spin-orbit_coupling = -8.3
orbital_reduction_factor =  0.328
high_symmetric_geometry-minimum_energy_geometry_distance = 0.1644
high_symmetric_geometry-saddle_point_geometry_distance = 0.1676

[excited_state_parameters]
Jahn-Teller_energy = 83.1466
barrier_energy = 6.7916
vibrational_energy_quantum = 75.6121
#spin-orbit coupling obtained from DFT calculation in meV:
DFT_spin-orbit_coupling = -95.9
orbital_reduction_factor =  0.782
high_symmetric_geometry-minimum_energy_geometry_distance = 0.3407
high_symmetric_geometry-saddle_point_geometry_distance = 0.3421



[magnetic_field]
minimum = 0.0
maximum = 10.0
direction_vector = 0.0, 0.0, 1.0
step_number = 11
#basis vector of the magnetic field
basis_vector_1 = 14.17860889, 0.0, 0.0
basis_vector_2 = 0.0, 14.17860889, 0.0
basis_vector_3 = 0.0, 0.0, 14.17860889
```

### 5. Four-State Model Configuration

For calculations using the four-state model:

```
[essentials]
maximum_number_of_vibrational_quanta = 12
output_prefix = SnV_ZPL_shift
output_folder = results/SnV_results
input_folder = data/SnV_data
save_raw_parameters = false
basis_vector_1 = 1, -1, 0
basis_vector_2 = 1, 1, -2
basis_vector_3 = 1, 1, 1
model_hamiltonian = false
save_model_hamiltonian_cfg = false

[ground_state_parameters]
dft_spin-orbit_coupling = -8.3
orbital_reduction_factor = 0.328
ham_reduction_factor = 0.4706
delta_p = 0.0470
k_jt = 0.0104

[excited_state_parameters]
dft_spin-orbit_coupling = -95.9
orbital_reduction_factor = 0.782
ham_reduction_factor = 0.1255
delta_p = 0.2967
k_jt = 0.0644

[magnetic_field]
minimum = 0.0
maximum = 10.0
step_number = 11
direction_vector = 1.0, 0.0, 0.0
basis_vector_1 = 14.17860889, 0.0, 0.0
basis_vector_2 = 0.0, 14.17860889, 0.0
basis_vector_3 = 0.0, 0.0, 14.17860889
```

## Output Options

The tool can generate various output files based on configuration:

- `save_raw_parameters = true`: Saves geometries in CSV format and generates a configuration file with atomic parameters
- `save_model_hamiltonian_cfg = true`: Generates model Hamiltonian configuration
- `save_taylor_coeffs_cfg = true`: Saves Taylor coefficients for Jahn-Teller interaction

## Eigenstates

The script can save eigenstates in either real or complex basis:

- Complex basis (e<sub>+</sub>, e<sub>-</sub>): Superpositions of degenerate orbitals (e<sub>x</sub>, e<sub>y</sub>)
  - e<sub>+</sub> = -(e<sub>x</sub>+ie<sub>y</sub>)
  - e<sub>-</sub> = e<sub>x</sub>-ie<sub>y</sub>

To use complex eigenstates, set:
```
eigen_states = complex
```

