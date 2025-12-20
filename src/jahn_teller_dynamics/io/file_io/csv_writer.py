"""
CSVWriter - General CSV file writer for various data types.

This class provides methods to write different types of data to CSV files:
- Eigen vectors and eigen values
- Eigen states
- Theoretical JT values
- Atomic coordinates
- General data tables

Can be integrated with PathManager for automatic folder path handling.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union, Any
import os

import jahn_teller_dynamics.physics.quantum_physics as qmp
import jahn_teller_dynamics.io.file_io.vasp as V
import jahn_teller_dynamics.math.matrix_mechanics as mm
# Lazy import to avoid circular dependencies
# results_formatter is imported inside functions that need it


class CSVWriter:
    """
    General CSV file writer for various data types.
    
    This class provides a unified interface for writing different types of
    scientific data to CSV files with consistent formatting.
    
    Can be integrated with PathManager to automatically use input/output folders
    and file prefixes from configuration.
    """
    
    def __init__(
        self,
        separator: str = ';',
        index: bool = True,
        path_manager: Optional[Any] = None
    ):
        """
        Initialize the CSV writer.
        
        Args:
            separator: CSV separator character (default: ';')
            index: Whether to include index in CSV output (default: True)
            path_manager: Optional PathManager instance for automatic path handling
        """
        self.separator = separator
        self.index = index
        self.path_manager = path_manager
    
    def _get_output_path(self, filename: str) -> str:
        """
        Get full output path using PathManager if available.
        
        Args:
            filename: Filename (with or without path)
            
        Returns:
            str: Full path with output folder prepended if PathManager available
        """
        if self.path_manager is None:
            return filename
        
        # If filename already has a path, use it as-is
        if os.path.dirname(filename):
            return filename
        
        # Otherwise, prepend output folder
        output_folder = self.path_manager.get_res_folder_name()
        return output_folder + filename if output_folder else filename
    
    def _get_input_path(self, filename: str) -> str:
        """
        Get full input path using PathManager if available.
        
        Args:
            filename: Filename (with or without path)
            
        Returns:
            str: Full path with input folder prepended if PathManager available
        """
        if self.path_manager is None:
            return filename
        
        # If filename already has a path, use it as-is
        if os.path.dirname(filename):
            return filename
        
        # Otherwise, prepend input folder
        input_folder = self.path_manager.get_data_folder_name()
        return input_folder + filename if input_folder else filename
    
    def _get_prefixed_filename(self, suffix: str) -> str:
        """
        Get filename with prefix from PathManager if available.
        
        Args:
            suffix: Filename suffix (e.g., '_eigen_vectors.csv')
            
        Returns:
            str: Filename with prefix prepended if PathManager available
        """
        if self.path_manager is None:
            return suffix.lstrip('_') if suffix.startswith('_') else suffix
        
        prefix = self.path_manager.get_prefix_name()+'_'
        if prefix:
            return prefix + suffix
        return suffix.lstrip('_') if suffix.startswith('_') else suffix
    
    # ==================== Eigen Vectors and Values ====================
    
    def write_eigen_vectors_and_values(
        self,
        eigen_vector_space: mm.eigen_vector_space,
        eigen_vec_filepath: str,
        eigen_val_filepath: str
    ) -> None:
        """
        Write eigen vectors and eigen values to CSV files.
        
        Args:
            eigen_vector_space: eigen_vector_space object containing eigen vectors
            eigen_vec_filepath: Path to save eigen vectors CSV
            eigen_val_filepath: Path to save eigen values CSV
        """
        eig_vec_df, eig_val_df = eigen_vector_space.create_eigen_kets_vals_table(
            eigen_vector_space.quantum_states_basis
        )
        
        eig_vec_df.to_csv(eigen_vec_filepath, sep=self.separator, index=self.index)
        eig_val_df.to_csv(eigen_val_filepath, sep=self.separator, index=self.index)
    
    def write_eigen_states(
        self,
        eigen_kets: List[mm.ket_vector],
        basis_states: List[str],
        eigen_vec_filepath: str,
        eigen_val_filepath: str,
        state_prefix: str = 'eigenstate_'
    ) -> None:
        """
        Write eigen states (vectors and values) to CSV files.
        
        Args:
            eigen_kets: List of ket_vector objects
            basis_states: List of basis state names/labels
            eigen_vec_filepath: Path to save eigen vectors CSV
            eigen_val_filepath: Path to save eigen values CSV
            state_prefix: Prefix for eigenstate names (default: 'eigenstate_')
        """
        eigen_kets_dict = {}
        eigen_kets_dict['basis_state'] = [str(state) for state in basis_states]
        
        eigen_vec_names = []
        eigen_vals = []
        
        for i, eigen_ket in enumerate(eigen_kets):
            eigen_vec_name = f'{state_prefix}{i+1}'
            eigen_vec_names.append(eigen_vec_name)
            eigen_vals.append(eigen_ket.eigen_val)
            eigen_kets_dict[eigen_vec_name] = eigen_ket.coeffs.tolist()
        
        eig_vecs_df = pd.DataFrame.from_dict(eigen_kets_dict)
        eig_vecs_df = eig_vecs_df.set_index('basis_state')
        
        eig_val_dict = {
            'state_name': eigen_vec_names,
            'eigenenergy': eigen_vals
        }
        eig_vals_df = pd.DataFrame.from_dict(eig_val_dict).set_index('state_name')
        
        eig_vecs_df.to_csv(eigen_vec_filepath, sep=self.separator, index=self.index)
        eig_vals_df.to_csv(eigen_val_filepath, sep=self.separator, index=self.index)
    
    # ==================== Atomic Coordinates ====================
    
    def write_atomic_coordinates(
        self,
        lattice: V.Lattice,
        filepath: str
    ) -> None:
        """
        Write atomic coordinates from a Lattice object to CSV.
        
        Args:
            lattice: Lattice object containing atomic coordinates
            filepath: Path to save the CSV file
        """
        df = lattice.to_coordinates_data_frame()
        df.to_csv(filepath, sep=self.separator, index=self.index)
    
    def write_atomic_coordinates_from_arrays(
        self,
        x_coords: List[float],
        y_coords: List[float],
        z_coords: List[float],
        filepath: str,
        index_name: str = 'index'
    ) -> None:
        """
        Write atomic coordinates from arrays to CSV.
        
        Args:
            x_coords: List of x coordinates
            y_coords: List of y coordinates
            z_coords: List of z coordinates
            filepath: Path to save the CSV file
            index_name: Name for the index column
        """
        if len(x_coords) != len(y_coords) or len(x_coords) != len(z_coords):
            raise ValueError("Coordinate arrays must have the same length")
        
        res_dict = {
            'x_coordinates': x_coords,
            'y_coordinates': y_coords,
            'z_coordinates': z_coords
        }
        
        res_df = pd.DataFrame(res_dict)
        res_df.index.name = index_name
        res_df.to_csv(filepath, sep=self.separator, index=self.index)
    
    # ==================== Theoretical JT Values ====================
    
    def write_theoretical_results(
        self,
        exe_tree: qmp.Exe_tree,
        filepath: str
    ) -> None:
        """
        Write theoretical JT results from Exe_tree to CSV.
        
        Args:
            exe_tree: Exe_tree object containing theoretical results
            filepath: Path to save the CSV file
        """
        # Lazy import to avoid circular dependencies - import directly from module
        from jahn_teller_dynamics.io.file_io.results_formatter import save_theoretical_results
        save_theoretical_results(exe_tree, filepath, separator=self.separator)
    
    def write_theoretical_results_from_dict(
        self,
        results_dict: Dict[str, Union[str, float, int]],
        filepath: str,
        attribute_col: str = 'attribute',
        value_col: str = 'values'
    ) -> None:
        """
        Write theoretical results from a dictionary to CSV.
        
        Args:
            results_dict: Dictionary with attribute names as keys and values as values
            filepath: Path to save the CSV file
            attribute_col: Name for the attribute column
            value_col: Name for the value column
        """
        res_dict = {
            attribute_col: list(results_dict.keys()),
            value_col: [str(v) for v in results_dict.values()]
        }
        res_df = pd.DataFrame(res_dict).set_index(attribute_col)
        res_df.to_csv(filepath, sep=self.separator, index=self.index)
    
    # ==================== General Data Tables ====================
    
    def write_dataframe(
        self,
        df: pd.DataFrame,
        filepath: str,
        index_col: Optional[str] = None
    ) -> None:
        """
        Write a pandas DataFrame to CSV.
        
        Args:
            df: DataFrame to write
            filepath: Path to save the CSV file
            index_col: Column name to use as index (if None, uses existing index)
        """
        if index_col is not None and index_col in df.columns:
            df = df.set_index(index_col)
        df.to_csv(filepath, sep=self.separator, index=self.index)
    
    def write_dict_to_csv(
        self,
        data_dict: Dict[str, List[Any]],
        filepath: str,
        index_col: Optional[str] = None
    ) -> None:
        """
        Write a dictionary to CSV (each key becomes a column).
        
        Args:
            data_dict: Dictionary with column names as keys and lists as values
            filepath: Path to save the CSV file
            index_col: Column name to use as index (if None, uses default index)
        """
        df = pd.DataFrame(data_dict)
        if index_col is not None and index_col in df.columns:
            df = df.set_index(index_col)
        df.to_csv(filepath, sep=self.separator, index=self.index)
    
    def write_transitions(
        self,
        transitions_dict: Dict[str, List[float]],
        filepath: str,
        index_col: str = 'magnetic field (T)'
    ) -> None:
        """
        Write transition energies to CSV (e.g., for ZPL calculations).
        
        Args:
            transitions_dict: Dictionary with transition labels and energy lists
            filepath: Path to save the CSV file
            index_col: Column name to use as index
        """
        df = pd.DataFrame(transitions_dict)
        if index_col in df.columns:
            df = df.set_index(index_col)
        df.to_csv(filepath, sep=self.separator, index=self.index)
    
    def write_energy_dependence(
        self,
        energy_dict: Dict[str, List[float]],
        field_strengths: List[float],
        filepath: str,
        field_col_name: str = 'magnetic field (T)'
    ) -> None:
        """
        Write energy dependence on field strength to CSV.
        
        Args:
            energy_dict: Dictionary with state labels and energy lists
            field_strengths: List of field strength values
            filepath: Path to save the CSV file
            field_col_name: Name for the field strength column
        """
        data_dict = {field_col_name: field_strengths}
        data_dict.update(energy_dict)
        df = pd.DataFrame(data_dict)
        df = df.set_index(field_col_name)
        df.to_csv(filepath, sep=self.separator, index=self.index)
    
    # ==================== PathManager-Integrated Methods ====================
    
    def write_eigen_vectors_and_values_to_output(
        self,
        eigen_vector_space: mm.eigen_vector_space,
        eigen_vec_suffix: str = '_eigen_vectors.csv',
        eigen_val_suffix: str = '_eigen_values.csv'
    ) -> tuple[str, str]:
        """
        Write eigen vectors and values to output folder with automatic naming.
        
        Args:
            eigen_vector_space: eigen_vector_space object
            eigen_vec_suffix: Suffix for eigen vectors file
            eigen_val_suffix: Suffix for eigen values file
            
        Returns:
            tuple: (eigen_vec_filepath, eigen_val_filepath)
        """
        eigen_vec_filename = self._get_prefixed_filename(eigen_vec_suffix)
        eigen_val_filename = self._get_prefixed_filename(eigen_val_suffix)
        
        eigen_vec_path = self._get_output_path(eigen_vec_filename)
        eigen_val_path = self._get_output_path(eigen_val_filename)
        
        # Ensure directories exist
        self.ensure_directory(eigen_vec_path)
        self.ensure_directory(eigen_val_path)
        
        self.write_eigen_vectors_and_values(
            eigen_vector_space,
            eigen_vec_path,
            eigen_val_path
        )
        
        return eigen_vec_path, eigen_val_path
    
    def write_atomic_coordinates_to_input(
        self,
        lattice: V.Lattice,
        filename: str
    ) -> str:
        """
        Write atomic coordinates to input folder.
        
        Args:
            lattice: Lattice object
            filename: Filename (will be placed in input folder)
            
        Returns:
            str: Full filepath used
        """
        filepath = self._get_input_path(filename)
        self.ensure_directory(filepath)
        self.write_atomic_coordinates(lattice, filepath)
        return filepath
    
    def write_theoretical_results_to_output(
        self,
        exe_tree: qmp.Exe_tree,
        suffix: str = '_theoretical_results.csv'
    ) -> str:
        """
        Write theoretical results to output folder with automatic naming.
        
        Args:
            exe_tree: Exe_tree object
            suffix: Filename suffix
            
        Returns:
            str: Full filepath used
        """
        filename = self._get_prefixed_filename(suffix)
        filepath = self._get_output_path(filename)
        self.ensure_directory(filepath)
        self.write_theoretical_results(exe_tree, filepath)
        return filepath
    
    def write_transitions_to_output(
        self,
        transitions_dict: Dict[str, List[float]],
        suffix: str,
        index_col: str = 'magnetic field (T)'
    ) -> str:
        """
        Write transitions to output folder with automatic naming.
        
        Args:
            transitions_dict: Dictionary with transition data
            suffix: Filename suffix (e.g., '_A_transitions.csv')
            index_col: Column name to use as index
            
        Returns:
            str: Full filepath used
        """
        filename = self._get_prefixed_filename(suffix)
        filepath = self._get_output_path(filename)
        self.ensure_directory(filepath)
        self.write_transitions(transitions_dict, filepath, index_col)
        return filepath
    
    def write_energy_dependence_to_output(
        self,
        energy_dict: Dict[str, List[float]],
        field_strengths: List[float],
        suffix: str = '_energy_vs_field.csv',
        field_col_name: str = 'magnetic field (T)'
    ) -> str:
        """
        Write energy dependence to output folder with automatic naming.
        
        Args:
            energy_dict: Dictionary with energy data
            field_strengths: List of field strength values
            suffix: Filename suffix
            field_col_name: Name for the field strength column
            
        Returns:
            str: Full filepath used
        """
        filename = self._get_prefixed_filename(suffix)
        filepath = self._get_output_path(filename)
        self.ensure_directory(filepath)
        self.write_energy_dependence(energy_dict, field_strengths, filepath, field_col_name)
        return filepath
    
    def write_LzSz_expected_values_to_output(
        self,
        lzsz_data: Union[pd.DataFrame, Dict[str, List[Any]]],
        filename_or_suffix: str = '_LzSz_expected_values.csv',
        index_col: str = 'state_name'
    ) -> str:
        """
        Write LzSz expected values to output folder with automatic naming.
        
        Args:
            lzsz_data: DataFrame or dictionary with LzSz data.
                       Expected columns: 'state_name', 'eigenenergy', 'LzSz'
            filename_or_suffix: Filename (with or without prefix) or suffix.
                                If it starts with '_', it's treated as a suffix and prefix is added.
                                Otherwise, it's used as-is (assumed to already include prefix if needed).
            index_col: Column name to use as index (default: 'state_name')
            
        Returns:
            str: Full filepath used
        """
        # Convert dict to DataFrame if needed
        if isinstance(lzsz_data, dict):
            df = pd.DataFrame(lzsz_data)
        else:
            df = lzsz_data.copy()
        
        # Set index if specified and column exists
        if index_col and index_col in df.columns:
            df = df.set_index(index_col)
        
        # Determine filename: if it starts with '_', treat as suffix and add prefix
        # Otherwise, use as-is (may already include prefix)
        if filename_or_suffix.startswith('_'):
            filename = self._get_prefixed_filename(filename_or_suffix)
        else:
            filename = filename_or_suffix
        
        filepath = self._get_output_path(filename)
        self.ensure_directory(filepath)
        df.to_csv(filepath, sep=self.separator, index=self.index)
        return filepath
    
    def set_path_manager(self, path_manager: Any) -> None:
        """
        Set or update the PathManager instance.
        
        Args:
            path_manager: PathManager instance
        """
        self.path_manager = path_manager
    
    # ==================== Utility Methods ====================
    
    def ensure_directory(self, filepath: str) -> None:
        """
        Ensure the directory for a filepath exists.
        
        Args:
            filepath: Full file path
        """
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
    
    def write_with_directory(
        self,
        df: pd.DataFrame,
        filepath: str,
        index_col: Optional[str] = None
    ) -> None:
        """
        Write DataFrame to CSV, creating directory if needed.
        
        Args:
            df: DataFrame to write
            filepath: Path to save the CSV file
            index_col: Column name to use as index
        """
        self.ensure_directory(filepath)
        self.write_dataframe(df, filepath, index_col)
    
    def create_LzSz_dataframe(
        self,
        LzSz_data: Dict[str, List[Any]],
        index_col: str = 'state_name'
    ) -> pd.DataFrame:
        """
        Create a DataFrame from LzSz data dictionary and set index.
        
        Args:
            LzSz_data: Dictionary with keys 'state_name', 'eigenenergy', 'LzSz'
            index_col: Column name to use as index (default: 'state_name')
            
        Returns:
            DataFrame with the specified column set as index
        """
        LzSz_res_df = pd.DataFrame(LzSz_data)
        LzSz_res_df = LzSz_res_df.set_index(index_col)
        return LzSz_res_df

