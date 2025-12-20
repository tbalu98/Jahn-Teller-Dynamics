"""
ConfigWriter - Writes configuration files from calculation results.

This class handles writing configuration files based on calculation results,
including saving raw parameters, model Hamiltonian configs, and Taylor coefficients.
"""

from configparser import ConfigParser
import os
from typing import Optional
import jahn_teller_dynamics.physics.quantum_physics as qmp

from jahn_teller_dynamics.io.config.reader import ConfigReader
from jahn_teller_dynamics.io.config.parameter_extractor import ParameterExtractor
from jahn_teller_dynamics.io.utils.path_manager import PathManager

# Import constants
from jahn_teller_dynamics.io.config.constants import (
    essentials_field, gnd_state_field, ex_state_field, single_case_section,
    atom_structure_field, mag_field,
    int_soc_opt, orb_red_fact_op, save_raw_pars_opt, model_Hamiltonian_opt,
    save_model_Hamiltonian_cfg_opt, save_Taylor_coeffs_cfg_opt,
    symm_latt_opt, JT_latt_opt, barr_latt_opt,
    symm_latt_energy_opt, min_energy_latt_energy_opt, saddle_point_latt_energy_opt,
    SOC_split_opt, delta_f_opt, f_factor_opt, F_opt, G_opt, hw_opt,
    min_field_opt, max_field_opt, step_num_opt, dir_vec_opt,
    basis_vector_1_opt, basis_vector_2_opt, basis_vector_3_opt
)


class ConfigWriter:
    """
    Writes configuration files from calculation results.
    
    This class provides methods to save configuration files based on
    calculation results, including raw parameters, model parameters,
    and Taylor coefficients.
    """
    
    def __init__(
        self,
        config_reader: ConfigReader,
        parameter_extractor: ParameterExtractor,
        path_manager: PathManager,
        original_config: ConfigParser,
        config_file_dir: str
    ):
        """
        Initialize the config writer.
        
        Args:
            config_reader: ConfigReader instance for reading config values
            parameter_extractor: ParameterExtractor instance for getting parameters
            path_manager: PathManager instance for path operations
            original_config: Original ConfigParser instance (to copy sections)
            config_file_dir: Directory where config files should be written
        """
        self.reader = config_reader
        self.params = parameter_extractor
        self.paths = path_manager
        self.original_config = original_config
        self.config_file_dir = config_file_dir
        # Directory for generated config files
        self.generated_config_dir = os.path.join(config_file_dir, 'generated_config_files')
        # Ensure the directory exists
        os.makedirs(self.generated_config_dir, exist_ok=True)
    
    def save_raw_pars(self, JT_int: qmp.Exe_tree) -> None:
        """
        Save raw parameters for a single case calculation.
        
        Args:
            JT_int: Exe_tree instance containing calculation results
        """
        if not self.reader.conditional_option(essentials_field, save_raw_pars_opt):
            return
        
        data_folder = self.paths.get_data_folder_name()
        problem_name = self.paths.get_prefix_name()
        
        new_config = ConfigParser()
        new_config[single_case_section] = self._save_raw_pars_section(JT_int, single_case_section)
        new_config[essentials_field] = self.original_config[essentials_field]
        new_config[essentials_field][save_raw_pars_opt] = 'false'
        
        new_config[atom_structure_field] = JT_int.JT_theory.symm_lattice.create_atom_pars_dict()
        
        self._add_magnetic_field_to_cfg(new_config, JT_int)
        
        csv_cfg_name = os.path.join(self.generated_config_dir, problem_name + '_csv_generated.cfg')
        with open(csv_cfg_name, 'w') as config_file:
            new_config.write(config_file)
    
    def save_model_pars(self, JT_int: qmp.Exe_tree) -> None:
        """
        Save model Hamiltonian parameters for a single case calculation.
        
        Args:
            JT_int: Exe_tree instance containing calculation results
        """
        if not self.reader.conditional_option(essentials_field, save_model_Hamiltonian_cfg_opt):
            return
        
        # Convert to minimal Exe_tree if needed
        if not isinstance(JT_int, qmp.minimal_Exe_tree):
            JT_int = qmp.minimal_Exe_tree.from_Exe_tree(JT_int)
        
        problem_name = self.paths.get_prefix_name()
        
        new_config = ConfigParser()
        new_config[single_case_section] = self._save_model_raw_pars_section(JT_int, single_case_section)
        new_config[essentials_field] = self.original_config[essentials_field]
        new_config[essentials_field][save_raw_pars_opt] = 'false'
        new_config[essentials_field][save_model_Hamiltonian_cfg_opt] = 'false'
        new_config[essentials_field][save_Taylor_coeffs_cfg_opt] = 'false'
        # Remove model_hamiltonian option if it exists - it should not be included in saved model config files
        if model_Hamiltonian_opt in new_config[essentials_field]:
            del new_config[essentials_field][model_Hamiltonian_opt]
        
        self._add_mag_field_to_cfg(new_config, JT_int)
        
        model_cfg_name = os.path.join(self.generated_config_dir, problem_name + '_model_generated.cfg')
        with open(model_cfg_name, 'w') as config_file:
            new_config.write(config_file)
    
    def save_raw_pars_ZPL(
        self,
        JT_int_gnd: qmp.Exe_tree,
        JT_int_ex: qmp.Exe_tree
    ) -> None:
        """
        Save raw parameters for a ZPL calculation (ground + excited states).
        
        Args:
            JT_int_gnd: Exe_tree for ground state
            JT_int_ex: Exe_tree for excited state
        """
        problem_name = self.paths.get_prefix_name()
        
        new_ZPL_config = ConfigParser()
        
        new_ZPL_config[gnd_state_field] = self._save_raw_pars_section(JT_int_gnd, gnd_state_field)
        new_ZPL_config[ex_state_field] = self._save_raw_pars_section(JT_int_ex, ex_state_field)
        new_ZPL_config[atom_structure_field] = JT_int_ex.JT_theory.symm_lattice.create_atom_pars_dict()
        
        new_ZPL_config[essentials_field] = self.original_config[essentials_field]
        new_ZPL_config[essentials_field][save_raw_pars_opt] = 'false'
        new_ZPL_config[essentials_field][model_Hamiltonian_opt] = 'false'
        new_ZPL_config[essentials_field][save_model_Hamiltonian_cfg_opt] = 'false'
        
        self._add_mag_field_to_cfg(new_ZPL_config, JT_int_gnd)
        
        csv_cfg_name = os.path.join(self.generated_config_dir, problem_name + '_csv_generated.cfg')
        with open(csv_cfg_name, 'w') as config_file:
            new_ZPL_config.write(config_file)
    
    def save_raw_pars_ZPL_model(
        self,
        JT_int_gnd: qmp.Exe_tree,
        JT_int_ex: qmp.Exe_tree
    ) -> None:
        """
        Save model Hamiltonian config for ZPL calculation.
        
        Args:
            JT_int_gnd: Exe_tree for ground state
            JT_int_ex: Exe_tree for excited state
        """
        # Convert to minimal Exe_tree if needed
        if not isinstance(JT_int_gnd, qmp.minimal_Exe_tree):
            JT_int_gnd = qmp.minimal_Exe_tree.from_Exe_tree(JT_int_gnd)
        if not isinstance(JT_int_ex, qmp.minimal_Exe_tree):
            JT_int_ex = qmp.minimal_Exe_tree.from_Exe_tree(JT_int_ex)
        
        problem_name = self.paths.get_prefix_name()
        
        new_ZPL_config = ConfigParser()
        new_ZPL_config[essentials_field] = self.original_config[essentials_field]
        new_ZPL_config[essentials_field][save_raw_pars_opt] = 'false'
        new_ZPL_config[essentials_field][save_model_Hamiltonian_cfg_opt] = 'false'
        new_ZPL_config[essentials_field][save_Taylor_coeffs_cfg_opt] = 'false'
        # Remove model_hamiltonian option if it exists - it should not be included in saved model config files
        if model_Hamiltonian_opt in new_ZPL_config[essentials_field]:
            del new_ZPL_config[essentials_field][model_Hamiltonian_opt]
        
        new_ZPL_config[gnd_state_field] = self._save_model_raw_pars_section(JT_int_gnd, gnd_state_field)
        new_ZPL_config[ex_state_field] = self._save_model_raw_pars_section(JT_int_ex, ex_state_field)
        
        self._add_mag_field_to_cfg(new_ZPL_config, JT_int_gnd)
        
        model_cfg_name = os.path.join(self.generated_config_dir, problem_name + '_model_generated.cfg')
        with open(model_cfg_name, 'w') as config_file:
            new_ZPL_config.write(config_file)
    
    def save_raw_pars_ZPL_Taylor(
        self,
        JT_int_gnd: qmp.Exe_tree,
        JT_int_ex: qmp.Exe_tree
    ) -> None:
        """
        Save Taylor coefficients config for ZPL calculation.
        
        Args:
            JT_int_gnd: Exe_tree for ground state
            JT_int_ex: Exe_tree for excited state
        """
        # Skip if already minimal Exe_tree
        if isinstance(JT_int_ex, qmp.minimal_Exe_tree) or isinstance(JT_int_gnd, qmp.minimal_Exe_tree):
            return
        
        problem_name = self.paths.get_prefix_name()
        
        new_ZPL_config = ConfigParser()
        new_ZPL_config[essentials_field] = self.original_config[essentials_field]
        new_ZPL_config[essentials_field][save_raw_pars_opt] = 'false'
        new_ZPL_config[essentials_field][model_Hamiltonian_opt] = 'false'
        new_ZPL_config[essentials_field][save_model_Hamiltonian_cfg_opt] = 'false'
        new_ZPL_config[essentials_field][save_Taylor_coeffs_cfg_opt] = 'false'
        
        new_ZPL_config[gnd_state_field] = self._save_Taylor_raw_pars_section(JT_int_gnd, gnd_state_field)
        new_ZPL_config[ex_state_field] = self._save_Taylor_raw_pars_section(JT_int_ex, ex_state_field)
        
        # Save magnetic field
        if self.reader.has_section(mag_field):
            new_ZPL_config.add_section(mag_field)
            new_ZPL_config[mag_field][min_field_opt] = self.original_config[mag_field][min_field_opt]
            new_ZPL_config[mag_field][max_field_opt] = self.original_config[mag_field][max_field_opt]
            new_ZPL_config[mag_field][step_num_opt] = self.original_config[mag_field][step_num_opt]
            new_ZPL_config[mag_field][dir_vec_opt] = self.original_config[mag_field][dir_vec_opt]
            
            # Add basis vectors from lattice if available
            if (JT_int_gnd.JT_theory.symm_lattice.basis_vecs is not None and
                len(JT_int_gnd.JT_theory.symm_lattice.basis_vecs) >= 3):
                new_ZPL_config[mag_field][basis_vector_1_opt] = str(JT_int_gnd.JT_theory.symm_lattice.basis_vecs[0])
                new_ZPL_config[mag_field][basis_vector_2_opt] = str(JT_int_gnd.JT_theory.symm_lattice.basis_vecs[1])
                new_ZPL_config[mag_field][basis_vector_3_opt] = str(JT_int_gnd.JT_theory.symm_lattice.basis_vecs[2])
        
        taylor_cfg_name = os.path.join(self.generated_config_dir, problem_name + '_Taylor_coeffs_generated.cfg')
        with open(taylor_cfg_name, 'w') as config_file:
            new_ZPL_config.write(config_file)
    
    def save_raw_pars_Taylor(self, JT_int: qmp.Exe_tree) -> None:
        """
        Save Taylor coefficients config for a single case calculation.
        
        Args:
            JT_int: Exe_tree instance containing calculation results
        """
        # Skip if already minimal Exe_tree
        if isinstance(JT_int, qmp.minimal_Exe_tree):
            return
        
        problem_name = self.paths.get_prefix_name()
        
        new_config = ConfigParser()
        new_config[essentials_field] = self.original_config[essentials_field]
        new_config[essentials_field][save_raw_pars_opt] = 'false'
        new_config[essentials_field][model_Hamiltonian_opt] = 'false'
        new_config[essentials_field][save_model_Hamiltonian_cfg_opt] = 'false'
        new_config[essentials_field][save_Taylor_coeffs_cfg_opt] = 'false'
        
        new_config[single_case_section] = self._save_Taylor_raw_pars_section(JT_int, single_case_section)
        
        self._add_mag_field_to_cfg(new_config, JT_int)
        
        taylor_cfg_name = os.path.join(self.generated_config_dir, problem_name + '_Taylor_coeffs_generated.cfg')
        with open(taylor_cfg_name, 'w') as config_file:
            new_config.write(config_file)
    
    # ==================== Helper Methods ====================
    
    def _save_raw_pars_section(self, JT_int: qmp.Exe_tree, section_to_cfg: str) -> dict:
        """
        Create a section dictionary with raw parameters from Exe_tree.
        
        Args:
            JT_int: Exe_tree instance
            section_to_cfg: Section name for the config
            
        Returns:
            dict: Dictionary of parameter name-value pairs
        """
        state_parameters_section = {}
        
        state_parameters_section[int_soc_opt] = str(self.params.get_spin_orbit_coupling(section_to_cfg))
        state_parameters_section[orb_red_fact_op] = str(self.params.get_gL_factor(section_to_cfg))
        
        problem_name = self.paths.get_prefix_name()
        symm_geom = JT_int.JT_theory.symm_lattice
        saddle_point_geom = JT_int.JT_theory.barrier_lattice if JT_int.JT_theory.barrier_lattice is not None else None
        min_energy_geom = JT_int.JT_theory.JT_lattice
        data_folder = self.paths.get_data_folder_name()
        
        # Save symmetric lattice
        df = symm_geom.to_coordinates_data_frame()
        filenamebase = problem_name + '_' + section_to_cfg
        symm_latt_geom_filename = filenamebase + '_' + symm_latt_opt + '.csv'
        df.to_csv(data_folder + symm_latt_geom_filename, sep=';')
        state_parameters_section[symm_latt_opt] = symm_latt_geom_filename
        state_parameters_section[symm_latt_energy_opt] = str(symm_geom.energy)
        
        # Save minimum energy lattice
        less_symm_latt_geom_filename_1 = filenamebase + '_' + JT_latt_opt + '.csv'
        df = min_energy_geom.to_coordinates_data_frame()
        df.to_csv(data_folder + less_symm_latt_geom_filename_1, sep=';')
        state_parameters_section[JT_latt_opt] = less_symm_latt_geom_filename_1
        state_parameters_section[min_energy_latt_energy_opt] = str(min_energy_geom.energy)
        
        # Save saddle point lattice if available
        if saddle_point_geom is not None:
            state_parameters_section[saddle_point_latt_energy_opt] = str(saddle_point_geom.energy)
            df = saddle_point_geom.to_coordinates_data_frame()
            barrier_lattice_filename = filenamebase + '_' + barr_latt_opt + '.csv'
            df.to_csv(data_folder + barrier_lattice_filename, sep=';')
            state_parameters_section[barr_latt_opt] = barrier_lattice_filename
        
        return state_parameters_section
    
    def _save_model_raw_pars_section(
        self,
        JT_int: qmp.minimal_Exe_tree,
        section_to_cfg: str
    ) -> dict:
        """
        Create a section dictionary with model Hamiltonian parameters.
        
        Args:
            JT_int: minimal_Exe_tree instance
            section_to_cfg: Section name for the config
            
        Returns:
            dict: Dictionary of parameter name-value pairs
        """
        state_parameters_section = {}
        
        # Get SOC split from lambda_theory
        state_parameters_section[SOC_split_opt] = str(
            JT_int.lambda_theory if JT_int.electron else -JT_int.lambda_theory
        )
        # Note: orbital_reduction_factor is NOT saved - it's an input parameter
        state_parameters_section[delta_f_opt] = str(JT_int.delta_f_factor)
        state_parameters_section[f_factor_opt] = str(JT_int.f_factor)
        
        return state_parameters_section
    
    def _save_Taylor_raw_pars_section(self, JT_int: qmp.Exe_tree, section_to_cfg: str) -> dict:
        """
        Create a section dictionary with Taylor coefficients.
        
        Args:
            JT_int: Exe_tree instance
            section_to_cfg: Section name for the config
            
        Returns:
            dict: Dictionary of parameter name-value pairs
        """
        state_parameters_section = {}
        
        state_parameters_section[int_soc_opt] = str(self.params.get_spin_orbit_coupling(section_to_cfg))
        state_parameters_section[orb_red_fact_op] = str(self.params.get_gL_factor(section_to_cfg))
        state_parameters_section[F_opt] = str(JT_int.JT_theory.F)
        state_parameters_section[G_opt] = str(JT_int.JT_theory.G)
        state_parameters_section[hw_opt] = str(JT_int.JT_theory.hw_meV)
        
        return state_parameters_section
    
    def _add_mag_field_to_cfg(
        self,
        new_config: ConfigParser,
        JT_int: qmp.Exe_tree
    ) -> ConfigParser:
        """
        Add magnetic field section to config.
        
        Args:
            new_config: ConfigParser to add magnetic field to
            JT_int: Exe_tree instance (for getting basis vectors if needed)
            
        Returns:
            ConfigParser: Updated config parser
        """
        if not self.reader.has_section(mag_field):
            return new_config
        
        new_config.add_section(mag_field)
        new_config[mag_field][min_field_opt] = self.original_config[mag_field][min_field_opt]
        new_config[mag_field][max_field_opt] = self.original_config[mag_field][max_field_opt]
        new_config[mag_field][step_num_opt] = self.original_config[mag_field][step_num_opt]
        new_config[mag_field][dir_vec_opt] = self.original_config[mag_field][dir_vec_opt]
        
        # Try to get basis vectors from config, fall back to lattice if available
        basis_vectors = self._get_basis_col_vectors_from_config(mag_field)
        if basis_vectors is not None:
            new_config[mag_field][basis_vector_1_opt] = str(basis_vectors[0])
            new_config[mag_field][basis_vector_2_opt] = str(basis_vectors[1])
            new_config[mag_field][basis_vector_3_opt] = str(basis_vectors[2])
        elif (JT_int.JT_theory.symm_lattice.basis_vecs is not None and
              len(JT_int.JT_theory.symm_lattice.basis_vecs) >= 3):
            new_config[mag_field][basis_vector_1_opt] = str(JT_int.JT_theory.symm_lattice.basis_vecs[0])
            new_config[mag_field][basis_vector_2_opt] = str(JT_int.JT_theory.symm_lattice.basis_vecs[1])
            new_config[mag_field][basis_vector_3_opt] = str(JT_int.JT_theory.symm_lattice.basis_vecs[2])
        
        return new_config
    
    def _add_magnetic_field_to_cfg(
        self,
        new_config: ConfigParser,
        JT_int: qmp.Exe_tree
    ) -> ConfigParser:
        """
        Add magnetic field section to config (simpler version without basis vectors).
        
        Args:
            new_config: ConfigParser to add magnetic field to
            JT_int: Exe_tree instance (unused, kept for compatibility)
            
        Returns:
            ConfigParser: Updated config parser
        """
        if self.reader.has_section(mag_field):
            new_config.add_section(mag_field)
            new_config[mag_field][min_field_opt] = self.original_config[mag_field][min_field_opt]
            new_config[mag_field][max_field_opt] = self.original_config[mag_field][max_field_opt]
            new_config[mag_field][step_num_opt] = self.original_config[mag_field][step_num_opt]
            new_config[mag_field][dir_vec_opt] = self.original_config[mag_field][dir_vec_opt]
        return new_config
    
    def _get_basis_col_vectors_from_config(self, field_name: str):
        """
        Get basis column vectors from config (helper method).
        
        Args:
            field_name: Section name
            
        Returns:
            List of basis vectors or None
        """
        import jahn_teller_dynamics.math.maths as maths
        
        b1_coords = self.reader.get_splitted_strs(field_name, basis_vector_1_opt, float)
        b2_coords = self.reader.get_splitted_strs(field_name, basis_vector_2_opt, float)
        b3_coords = self.reader.get_splitted_strs(field_name, basis_vector_3_opt, float)
        
        if b1_coords is None or b2_coords is None or b3_coords is None:
            return None
        
        b1 = maths.col_vector.from_list(b1_coords)
        b2 = maths.col_vector.from_list(b2_coords)
        b3 = maths.col_vector.from_list(b3_coords)
        
        return [b1.normalize(), b2.normalize(), b3.normalize()]

