"""
AtomConfigParser - Parser for atom structure and basis vector information.

This class provides a simple interface for parsing atom structure data
and basis vectors from configuration files. It's used by the physics
module to build JT theory from CSV files and parameters.
"""

from typing import List, Optional
import jahn_teller_dynamics.math.maths as maths
from jahn_teller_dynamics.io.config.reader import ConfigReader
from jahn_teller_dynamics.io.config.parameter_extractor import ParameterExtractor
from jahn_teller_dynamics.io.config.field_parser import FieldVectorParser

# Import constants
from jahn_teller_dynamics.io.config.constants import (
    atom_structure_field,
    basis_vector_1_opt, basis_vector_2_opt, basis_vector_3_opt,
    names_of_atoms_op, mass_of_atoms_op, num_of_atoms_op,
    symm_latt_energy_opt, min_energy_latt_energy_opt, saddle_point_latt_energy_opt
)


class AtomConfigParser:
    """
    Parser for atom structure and basis vector information from config files.
    
    This class provides methods to extract:
    - Basis vectors
    - Atom names, masses, and numbers
    - Lattice energies
    """
    
    def __init__(self, config_file_path: str):
        """
        Initialize the atom config parser.
        
        Args:
            config_file_path: Path to the configuration file
        """
        from configparser import ConfigParser
        
        # Read config file
        with open(config_file_path, 'r') as config_file:
            config_string = config_file.read()
        
        self.config = ConfigParser()
        self.config.read_string(config_string)
        
        # Initialize components
        self.reader = ConfigReader(self.config)
        self.params = ParameterExtractor(self.reader)
        self.fields = FieldVectorParser(self.reader)
    
    def get_basis_vectors(self) -> List[maths.col_vector]:
        """
        Get basis vectors from configuration.
        
        Uses the default atom_structure_field section.
        
        Returns:
            List of three normalized basis vectors [b1, b2, b3]
        """
        basis_vecs = self.fields.get_basis_col_vectors(atom_structure_field)
        if basis_vecs is None:
            return None
        return basis_vecs
    
    def get_names(self) -> List[str]:
        """
        Get atom names from configuration.
        
        Uses the default atom_structure_field section.
        
        Returns:
            List of atom names
        """
        return self.params.get_atom_names(atom_structure_field)
    
    def get_masses(self) -> List[float]:
        """
        Get atom masses from configuration.
        
        Uses the default atom_structure_field section.
        
        Returns:
            List of atom masses
        """
        return self.params.get_masses(atom_structure_field)
    
    def get_numbers(self) -> List[int]:
        """
        Get atom numbers from configuration.
        
        Uses the default atom_structure_field section.
        
        Returns:
            List of atom numbers
        """
        return self.params.get_numbers(atom_structure_field)
    
    def get_lattice_energy(self, energy_section_name: str) -> float:
        """
        Get lattice energy from configuration.
        
        Maps energy section names to their corresponding options:
        - 'symm_lattice_energy' → uses symm_latt_energy_opt in atom_structure_field
        - 'JT_lattice_energy' → uses min_energy_latt_energy_opt in atom_structure_field
        - 'barrier_lattice_energy' → uses saddle_point_latt_energy_opt in atom_structure_field
        
        Args:
            energy_section_name: Energy section identifier (e.g., 'symm_lattice_energy')
            
        Returns:
            Lattice energy value
        """
        # Map energy section names to energy options
        energy_option_map = {
            'symm_lattice_energy': symm_latt_energy_opt,
            'JT_lattice_energy': min_energy_latt_energy_opt,
            'barrier_lattice_energy': saddle_point_latt_energy_opt,
        }
        
        if energy_section_name in energy_option_map:
            energy_opt = energy_option_map[energy_section_name]
            return self.params.get_lattice_energy(atom_structure_field, energy_opt)
        else:
            # Try to use the section name directly as an option
            return self.params.get_lattice_energy(atom_structure_field, energy_section_name)

