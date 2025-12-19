"""
SectionTypeDetector - Detects input source types and calculation modes.

This class provides methods to detect what type of input data is specified
in a configuration section, and what type of calculation is being performed.
"""

from jahn_teller_dynamics.io.config.reader import ConfigReader

# Import constants
from jahn_teller_dynamics.io.config.constants import (
    essentials_field, gnd_state_field, ex_state_field, single_case_section,
    symm_latt_opt, EJT_opt, F_opt, hw_opt, Ham_red_opt, SOC_split_opt,
    con_int_en_opt, eigen_states_opt, save_raw_pars_opt,
    save_model_Hamiltonian_cfg_opt, save_Taylor_coeffs_cfg_opt,
    model_Hamiltonian_opt
)


class SectionTypeDetector:
    """
    Detects input source types and calculation modes from configuration.
    
    This class provides methods to determine:
    - What type of input data is specified (XML, CSV, Taylor coeffs, etc.)
    - What type of calculation is being performed (ZPL, single case, etc.)
    - What options are enabled (save raw pars, use model Hamiltonian, etc.)
    """
    
    def __init__(self, config_reader: ConfigReader):
        """
        Initialize the section type detector.
        
        Args:
            config_reader: ConfigReader instance for reading config values
        """
        self.reader = config_reader
    
    # ==================== Input Source Type Detection ====================
    
    def is_from_JT_pars(self, section: str) -> bool:
        """
        Check if section specifies JT parameters (E_JT, delta, hw, distances).
        
        Args:
            section: Configuration section name
            
        Returns:
            bool: True if section contains EJT_opt
        """
        return self.reader.has_option(section, EJT_opt)
    
    def is_from_vasprun_xml(self, section: str) -> bool:
        """
        Check if section specifies VASP XML files (vasprun.xml).
        
        Args:
            section: Configuration section name
            
        Returns:
            bool: True if high_symmetry_geometry option ends with '.xml'
        """
        high_symm_latt_fn = self.reader.get_option_of_field(section, symm_latt_opt)
        if not high_symm_latt_fn:
            return False
        return high_symm_latt_fn.endswith('.xml')
    
    def is_from_csv(self, section: str) -> bool:
        """
        Check if section specifies CSV files.
        
        Args:
            section: Configuration section name
            
        Returns:
            bool: True if high_symmetry_geometry option ends with '.csv'
        """
        high_symm_latt_fn = self.reader.get_option_of_field(section, symm_latt_opt)
        if not high_symm_latt_fn:
            return False
        return high_symm_latt_fn.endswith('.csv')
    
    def is_from_Taylor_coeffs(self, section: str) -> bool:
        """
        Check if section specifies Taylor coefficients (F, G, hw).
        
        Args:
            section: Configuration section name
            
        Returns:
            bool: True if section contains both F_opt and hw_opt
        """
        return (self.reader.has_option(section, F_opt) and 
                self.reader.has_option(section, hw_opt))
    
    def is_from_model_Hamiltonian(self, section: str) -> bool:
        """
        Check if section specifies model Hamiltonian parameters.
        
        Args:
            section: Configuration section name
            
        Returns:
            bool: True if section contains Ham_red_opt or SOC_split_opt
        """
        return (self.reader.has_option(section, Ham_red_opt) or 
                self.reader.has_option(section, SOC_split_opt))
    
    def is_from_energy_distance_pairs(self, section: str) -> bool:
        """
        Check if section specifies energy-distance pairs.
        
        Args:
            section: Configuration section name
            
        Returns:
            bool: True if section contains con_int_en_opt
        """
        return self.reader.has_option(section, con_int_en_opt)
    
    # ==================== Calculation Type Detection ====================
    
    def is_ZPL_calculation(self) -> bool:
        """
        Check if this is a ZPL (Zero Phonon Line) calculation.
        
        A ZPL calculation requires both ground_state_parameters and
        excited_state_parameters sections.
        
        Returns:
            bool: True if both ground and excited state sections exist
        """
        return (self.reader.has_section(gnd_state_field) and 
                self.reader.has_section(ex_state_field))
    
    def is_single_case(self) -> bool:
        """
        Check if this is a single case calculation.
        
        Returns:
            bool: True if system_parameters section exists
        """
        return self.reader.has_section(single_case_section)
    
    # ==================== Option Flag Detection ====================
    
    def is_save_raw_pars(self) -> bool:
        """
        Check if raw parameters should be saved.
        
        Returns:
            bool: True if save_raw_pars_opt is set to 'true' in essentials
        """
        return self.reader.conditional_option(essentials_field, save_raw_pars_opt)
    
    def is_save_model_Hamiltonian_cfg(self) -> bool:
        """
        Check if model Hamiltonian config should be saved.
        
        Returns:
            bool: True if save_model_Hamiltonian_cfg_opt is set to 'true' in essentials
        """
        return self.reader.conditional_option(essentials_field, save_model_Hamiltonian_cfg_opt)
    
    def is_save_Taylor_coeffs_cfg(self) -> bool:
        """
        Check if Taylor coefficients config should be saved.
        
        Returns:
            bool: True if save_Taylor_coeffs_cfg_opt is set to 'true' in essentials
        """
        return self.reader.conditional_option(essentials_field, save_Taylor_coeffs_cfg_opt)
    
    def is_use_model_hamiltonian(self) -> bool:
        """
        Check if model Hamiltonian should be used.
        
        Returns:
            bool: True if model_Hamiltonian_opt is set to 'true' in essentials
        """
        return self.reader.conditional_option(essentials_field, model_Hamiltonian_opt)
    
    # ==================== Eigenstate Type Detection ====================
    
    def is_real_eigen_vects(self) -> bool:
        """
        Check if real eigenstates are requested.
        
        Returns:
            bool: True if eigen_states_opt is set to 'real' in essentials
        """
        eigen_type = self.reader.get_option_of_field(essentials_field, eigen_states_opt)
        return eigen_type == 'real'
    
    def is_complex_eigen_vects(self) -> bool:
        """
        Check if complex eigenstates are requested.
        
        Returns:
            bool: True if eigen_states_opt is set to 'complex' in essentials
        """
        eigen_type = self.reader.get_option_of_field(essentials_field, eigen_states_opt)
        return eigen_type == 'complex'

