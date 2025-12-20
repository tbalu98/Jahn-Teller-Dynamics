"""
Configuration Constants - Centralized definition of all configuration file constants.

This module defines all section names, option names, and other constants used
throughout the Jahn-Teller configuration file parsing system.

All constants are organized by category for better maintainability.
"""

# ==================== Section Names (Fields) ====================

# Main sections
essentials_field = 'essentials'
so_c_field = 'spin_orbit_coupling'
mag_field = 'magnetic_field'
csv_field = '.csv_files'
xml_field = 'vasprun.xml_files'
el_field = 'electric_field'
strain_field = 'strain_field'
atom_structure_field = 'atom_structure_parameters'

# State sections
ex_state_field = 'excited_state_parameters'
gnd_state_field = 'ground_state_parameters'
system_field = 'system_parameters'
single_case_section = 'system_parameters'  # Alias for backward compatibility

# ==================== Option Names ====================

# Path and folder options
out_folder_opt = 'output_folder'
in_folder_opt = 'input_folder'
out_prefix_opt = 'output_prefix'

# Physics parameter options
int_soc_opt = 'DFT_spin-orbit_coupling'
orb_red_fact_op = 'orbital_reduction_factor'
spectrum_range_opt = 'LS_spectrum_range'
max_vib_quant = 'maximum_number_of_vibrational_quanta'

# Jahn-Teller energy parameters
EJT_opt = 'Jahn-Teller_energy'
E_barr_opt = 'barrier_energy'
hw_opt = 'vibrational_energy_quantum'

# Taylor coefficient options
F_opt = 'F'
G_opt = 'G'

# Model Hamiltonian options
Ham_red_opt = 'Ham_reduction_factor'
SOC_split_opt = 'spin-orbit_splitting_energy'
f_factor_opt = 'f'
delta_f_opt = 'delta_f'
delta_p_opt = 'delta_p'
Yx_opt = 'Yx'
Yy_opt = 'Yy'
K_JT_opt = 'K_JT'

# Lattice geometry options
symm_latt_opt = 'high_symmetry_geometry'
JT_latt_opt = 'global_minimum_geometry'
barr_latt_opt = 'saddle_point_geometry'

# Lattice energy options
symm_latt_energy_opt = 'high_symmetric_geometry_energy'
min_energy_latt_energy_opt = 'global_minimum_energy'
saddle_point_latt_energy_opt = 'saddle_point_energy'

# Distance options
symm_min_dist_opt = 'high_symmetric_geometry-minimum_energy_geometry_distance'
symm_saddl_dist_opt = 'high_symmetric_geometry-saddle_point_geometry_distance'

# Energy-distance pair options
con_int_en_opt = 'conical_intersection_energy'
con_int_loc_opt = 'conical_intersection_location'
global_min_loc_opt = 'global_minimum_location'
saddle_point_loc_opt = 'saddle_point_location'

# Basis vector options
basis_vector_1_opt = 'basis_vector_1'
basis_vector_2_opt = 'basis_vector_2'
basis_vector_3_opt = 'basis_vector_3'

# Atom structure options
at_pat_opt = 'atom_parameters'
num_of_atoms_op = 'numbers_of_atoms'
mass_of_atoms_op = 'masses_of_atoms'
names_of_atoms_op = 'names_of_atoms'

# Magnetic field options
dir_opt = 'direction vector'
dir_vec_opt = 'direction_vector'
min_field_opt = 'minimum'
max_field_opt = 'maximum'
step_num_opt = 'step_number'
from_opt = 'from'
to_opt = 'to'

# Strain field options
strain_vec_op = 'strain_vector'

# Orientation options
orientation_vector_base = 'orientation_vector'

# Eigenstate options
eigen_states_opt = 'orbital_basis_states'

# Model and save options
model_Hamiltonian_opt = 'model_Hamiltonian'
save_raw_pars_opt = 'save_raw_parameters'
save_model_Hamiltonian_cfg_opt = 'save_model_Hamiltonian_cfg'
save_Taylor_coeffs_cfg_opt = 'save_taylor_coeffs_cfg'

# ==================== CSV Column Names ====================

mag_field_strength_csv_col = 'magnetic field (T)'

