"""
ParameterExtractor - Extracts physics parameters from configuration files.

This class provides methods to extract physics-related parameters from config sections,
such as spin-orbit coupling, gL factors, Taylor coefficients, atom properties, etc.
"""

from typing import Optional, List
from jahn_teller_dynamics.io.config.reader import ConfigReader

# Import constants from the dedicated constants module
from jahn_teller_dynamics.io.config.constants import (
    essentials_field, so_c_field,
    int_soc_opt, orb_red_fact_op, max_vib_quant, spectrum_range_opt,
    F_opt, G_opt, hw_opt, f_factor_opt, delta_f_opt, delta_p_opt,
    Yx_opt, Yy_opt, K_JT_opt, Ham_red_opt, SOC_split_opt,
    num_of_atoms_op, mass_of_atoms_op, names_of_atoms_op,
    symm_latt_energy_opt, min_energy_latt_energy_opt,
    saddle_point_latt_energy_opt, eigen_states_opt
)


class ParameterExtractor:
    """
    Extracts physics parameters from configuration sections.
    
    This class provides a high-level interface for extracting physics-related
    parameters, delegating low-level config reading to ConfigReader.
    """
    
    def __init__(self, config_reader: ConfigReader):
        """
        Initialize the parameter extractor.
        
        Args:
            config_reader: ConfigReader instance for reading config values
        """
        self.reader = config_reader
    
    # ==================== Spin-Orbit Coupling Parameters ====================
    
    def get_spin_orbit_coupling(self, section: str) -> float:
        """
        Get spin-orbit coupling value from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            float: Spin-orbit coupling value, or 0.0 if not found
        """
        value = self.reader.get_option_of_field(section, int_soc_opt)
        return float(value) if value != '' else 0.0
    
    def get_SOC_split(self, section: str) -> Optional[float]:
        """
        Get spin-orbit splitting energy from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            float: SOC splitting energy, or None if not found
        """
        return self.reader.get_float_option_of_field(section, SOC_split_opt)
    
    def get_gL_factor(self, section: str = so_c_field) -> float:
        """
        Get orbital reduction factor (gL factor) from a section.
        
        Args:
            section: Configuration section name (default: 'spin_orbit_coupling')
            
        Returns:
            float: gL factor, or 0.0 if not found
        """
        if self.reader.has_option(section, orb_red_fact_op):
            value = self.reader.get_option_of_field(section, orb_red_fact_op)
            return float(value) if value != '' else 0.0
        return 0.0
    
    # ==================== Vibrational Parameters ====================
    
    def get_order(self) -> int:
        """
        Get maximum number of vibrational quanta from essentials section.
        
        Returns:
            int: Maximum number of vibrational quanta, or 0 if not found
        """
        return self.reader.get_int_option_of_field(essentials_field, max_vib_quant)
    
    def get_hw(self, section: str) -> float:
        """
        Get vibrational energy quantum (hw) from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            float: Vibrational energy quantum
        """
        value = self.reader.get_option_of_field(section, hw_opt)
        return float(value) if value != '' else 0.0
    
    # ==================== Taylor Coefficients ====================
    
    def get_F_coeff(self, section: str) -> float:
        """
        Get F coefficient (first-order JT coupling) from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            float: F coefficient
        """
        value = self.reader.get_option_of_field(section, F_opt)
        return float(value) if value != '' else 0.0
    
    def get_G_coeff(self, section: str) -> float:
        """
        Get G coefficient (second-order JT coupling) from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            float: G coefficient
        """
        value = self.reader.get_option_of_field(section, G_opt)
        return float(value) if value != '' else 0.0
    
    # ==================== Model Hamiltonian Parameters ====================
    
    def get_p_factor(self, section: str) -> Optional[float]:
        """
        Get Hamiltonian reduction factor (p factor) from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            float: p factor, or None if not found
        """
        return self.reader.get_float_option_of_field(section, Ham_red_opt)
    
    def get_f_factor(self, section: str) -> Optional[float]:
        """
        Get f factor from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            float: f factor, or None if not found
        """
        return self.reader.get_float_option_of_field(section, f_factor_opt)
    
    def get_delta_f(self, section: str) -> float:
        """
        Get delta_f parameter from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            float: delta_f value, or 0.0 if not found
        """
        value = self.reader.get_option_of_field(section, delta_f_opt)
        return float(value) if value != '' else 0.0
    
    def get_delta_p(self, section: str) -> float:
        """
        Get delta_p parameter from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            float: delta_p value, or 0.0 if not found
        """
        value = self.reader.get_option_of_field(section, delta_p_opt)
        return float(value) if value != '' else 0.0
    
    def get_Yx(self, section: str) -> Optional[float]:
        """
        Get Yx parameter from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            float: Yx value, or None if not found
        """
        return self.reader.get_float_option_of_field(section, Yx_opt)
    
    def get_Yy(self, section: str) -> Optional[float]:
        """
        Get Yy parameter from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            float: Yy value, or None if not found
        """
        return self.reader.get_float_option_of_field(section, Yy_opt)
    
    def get_KJT(self, section: str) -> Optional[float]:
        """
        Get K_JT parameter from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            float: K_JT value, or None if not found
        """
        return self.reader.get_float_option_of_field(section, K_JT_opt)
    
    # ==================== Atom Structure Parameters ====================
    
    def get_numbers(self, section: str) -> List[int]:
        """
        Get list of atom numbers from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            List[int]: List of atom numbers
        """
        value = self.reader.get_option_of_field(section, num_of_atoms_op)
        if value == '':
            return []
        return [int(r.strip()) for r in value.split(',') if r.strip()]
    
    def get_atom_names(self, section: str) -> List[str]:
        """
        Get list of atom names from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            List[str]: List of atom names
        """
        value = self.reader.get_option_of_field(section, names_of_atoms_op)
        if value == '':
            return []
        return [name.strip() for name in value.split(',') if name.strip()]
    
    def get_masses(self, section: str) -> List[float]:
        """
        Get list of atom masses from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            List[float]: List of atom masses
        """
        value = self.reader.get_option_of_field(section, mass_of_atoms_op)
        if value == '':
            return []
        return [float(r.strip()) for r in value.split(',') if r.strip()]
    
    # ==================== Lattice Energy Parameters ====================
    
    def get_lattice_energy(self, section: str, energy_opt: str) -> float:
        """
        Get lattice energy from a section.
        
        Args:
            section: Configuration section name
            energy_opt: Option name for the energy (e.g., 'high_symmetric_geometry_energy')
            
        Returns:
            float: Lattice energy value
        """
        value = self.reader.get_option_of_field(section, energy_opt)
        return float(value) if value != '' else 0.0
    
    # ==================== Calculation Control Parameters ====================
    
    def get_calc_LzSz(self) -> int:
        """
        Get number of LzSz expectation values to calculate from essentials section.
        
        Returns:
            int: Number of LzSz calculations, or 0 if not found
        """
        value = self.reader.get_option_of_field(essentials_field, spectrum_range_opt)
        return int(value) if value != '' else 0
    
    def get_eigen_state_type(self) -> str:
        """
        Get eigenstate type (real or complex) from essentials section.
        
        Returns:
            str: 'real' or 'complex', defaults to 'real'
        """
        if self.reader.has_option(essentials_field, eigen_states_opt):
            value = self.reader.get_option_of_field(essentials_field, eigen_states_opt)
            return value if value != '' else 'real'
        return 'real'

