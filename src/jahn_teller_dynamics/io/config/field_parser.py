"""
FieldVectorParser - Parses and constructs field vectors from configuration.

This class handles parsing of field vectors (magnetic, electric, strain) and
basis vectors from configuration files.
"""

from typing import Optional, List
import numpy as np
import jahn_teller_dynamics.math.maths as maths

from jahn_teller_dynamics.io.config.reader import ConfigReader

# Import constants
from jahn_teller_dynamics.io.config.constants import (
    mag_field, strain_field, essentials_field,
    dir_vec_opt, strain_vec_op, min_field_opt, max_field_opt, step_num_opt,
    basis_vector_1_opt, basis_vector_2_opt, basis_vector_3_opt
)


class FieldVectorParser:
    """
    Parses and constructs field vectors from configuration.
    
    This class provides methods to parse field vectors (magnetic, electric, strain)
    and basis vectors from configuration sections.
    """
    
    def __init__(self, config_reader: ConfigReader):
        """
        Initialize the field vector parser.
        
        Args:
            config_reader: ConfigReader instance for reading config values
        """
        self.reader = config_reader
    
    # ==================== Magnetic Field Vectors ====================
    
    def get_magnetic_field_vectors(self) -> Optional[List[maths.col_vector]]:
        """
        Get list of magnetic field vectors from configuration.
        
        Returns:
            List[maths.col_vector]: List of magnetic field vectors, or None if no magnetic field section
        """
        if not self.reader.has_section(mag_field):
            return None
        
        Bs = self.get_mag_field_strengths_list()
        basis_vectors = self.get_basis_col_vectors(mag_field)
        dir_vec = self.get_mag_dir_vector()
        
        if dir_vec is None:
            return None
        
        dir_vec = dir_vec.normalize()
        B_fields = [B * dir_vec for B in Bs]
        
        if basis_vectors is not None:
            B_fields = [B_field.in_new_basis(basis_vectors) for B_field in B_fields]
        
        return B_fields
    
    def get_mag_dir_vector(self) -> Optional[maths.col_vector]:
        """
        Get magnetic field direction vector.
        
        Returns:
            maths.col_vector: Direction vector, or None if not found
        """
        coordinates = self.reader.get_splitted_strs(mag_field, dir_vec_opt, float)
        if coordinates is None:
            return None
        return maths.col_vector.from_list(coordinates)
    
    def get_mag_field_strengths_list(self) -> np.ndarray:
        """
        Get list of magnetic field strengths.
        
        Returns:
            np.ndarray: Array of field strengths
        """
        B_min = self.get_B_min()
        B_max = self.get_B_max()
        step_num = self.get_step_num()
        
        return np.linspace(B_min, B_max, step_num)
    
    def get_B_min(self) -> float:
        """Get minimum magnetic field strength."""
        value = self.reader.get_option_of_field(mag_field, min_field_opt)
        return float(value) if value else 0.0
    
    def get_B_max(self) -> float:
        """Get maximum magnetic field strength."""
        value = self.reader.get_option_of_field(mag_field, max_field_opt)
        return float(value) if value else 0.0
    
    def get_step_num(self) -> int:
        """Get number of steps for magnetic field."""
        return self.reader.get_int_option_of_field(mag_field, step_num_opt)
    
    # ==================== General Field Vectors ====================
    
    def get_field_vectors(self, field_name: str) -> Optional[List[maths.col_vector]]:
        """
        Get list of field vectors for a given field name.
        
        Args:
            field_name: Name of the field section
            
        Returns:
            List[maths.col_vector]: List of field vectors, or None if section doesn't exist
        """
        if not self.reader.has_section(field_name):
            return None
        
        field_strengths = self.get_field_strengths_list(field_name)
        basis_vectors = self.get_basis_col_vectors(field_name)
        dir_vec = self.get_field_dir(field_name)
        
        if dir_vec is None:
            return None
        
        dir_vec = dir_vec.normalize()
        field_vecs = [F * dir_vec for F in field_strengths]
        
        if basis_vectors is not None:
            field_vecs = [field_vec.in_new_basis(basis_vectors) for field_vec in field_vecs]
        
        return field_vecs
    
    def get_field_dir(self, field_name: str) -> Optional[maths.col_vector]:
        """
        Get field direction vector for a given field.
        
        Args:
            field_name: Name of the field section
            
        Returns:
            maths.col_vector: Direction vector, or None if not found
        """
        coordinates = self.reader.get_splitted_strs(field_name, dir_vec_opt, float)
        if coordinates is None:
            return None
        return maths.col_vector.from_list(coordinates)
    
    def get_field_strengths_list(self, field_name: str) -> np.ndarray:
        """
        Get list of field strengths for a given field.
        
        Args:
            field_name: Name of the field section
            
        Returns:
            np.ndarray: Array of field strengths
        """
        min_val = self.get_field_min(field_name)
        max_val = self.get_field_max(field_name)
        step_num = self.get_field_step_num(field_name)
        
        return np.linspace(min_val, max_val, step_num)
    
    def get_field_min(self, field_name: str) -> float:
        """Get minimum field strength."""
        value = self.reader.get_option_of_field(field_name, min_field_opt)
        return float(value) if value else 0.0
    
    def get_field_max(self, field_name: str) -> float:
        """Get maximum field strength."""
        value = self.reader.get_option_of_field(field_name, max_field_opt)
        return float(value) if value else 0.0
    
    def get_field_step_num(self, field_name: str) -> int:
        """Get number of steps for field."""
        return self.reader.get_int_option_of_field(field_name, step_num_opt)
    
    # ==================== Strain Field ====================
    
    def get_strain_field_vector(self) -> Optional[maths.col_vector]:
        """
        Get strain field vector.
        
        Returns:
            maths.col_vector: Strain field vector, or None if not found
        """
        if not self.reader.has_section(strain_field):
            return None
        return self.get_strain_dir_vector()
    
    def get_strain_dir_vector(self) -> Optional[maths.col_vector]:
        """
        Get strain field direction vector.
        
        Returns:
            maths.col_vector: Strain direction vector, or None if not found
        """
        coordinates = self.reader.get_splitted_strs(strain_field, strain_vec_op, float)
        if coordinates is None:
            return None
        return maths.col_vector.from_list(coordinates)
    
    # ==================== Basis Vectors ====================
    
    def get_basis_col_vectors(self, field_name: str) -> Optional[List[maths.col_vector]]:
        """
        Get basis column vectors for a field.
        
        Args:
            field_name: Name of the field section
            
        Returns:
            List[maths.col_vector]: List of three normalized basis vectors, or None if not found
        """
        b1 = self.get_col_vector(field_name, basis_vector_1_opt)
        b2 = self.get_col_vector(field_name, basis_vector_2_opt)
        b3 = self.get_col_vector(field_name, basis_vector_3_opt)
        
        if b1 is None or b2 is None or b3 is None:
            return None
        
        return [b1.normalize(), b2.normalize(), b3.normalize()]
    
    def get_col_vector(self, field_name: str, opt_name: str) -> Optional[maths.col_vector]:
        """
        Get a column vector from a field option.
        
        Args:
            field_name: Name of the field section
            opt_name: Option name for the vector
            
        Returns:
            maths.col_vector: Column vector, or None if not found
        """
        coordinates = self.reader.get_splitted_strs(field_name, opt_name, float)
        if coordinates is None:
            return None
        return maths.col_vector.from_list(coordinates)
    
    def get_system_orientation_basis(self) -> Optional[List[maths.col_vector]]:
        """
        Get system orientation basis vectors from essentials section.
        
        Returns:
            List[maths.col_vector]: List of three normalized basis vectors, or None if not found
        """
        return self.get_basis_col_vectors(essentials_field)

