"""
PathManager - Manages file paths, folders, and prefixes from configuration.

This class handles all path-related operations, including input/output folders
and file naming prefixes.
"""

import os
from typing import Optional
from jahn_teller_dynamics.io.config.reader import ConfigReader

# Import constants
from jahn_teller_dynamics.io.config.constants import (
    essentials_field, out_folder_opt, in_folder_opt, out_prefix_opt
)


class PathManager:
    """
    Manages file paths, folders, and prefixes from configuration.
    
    This class provides methods to get input/output folder paths and
    output file prefixes from the configuration file.
    """
    
    def __init__(self, config_reader: ConfigReader, config_file_path: Optional[str] = None):
        """
        Initialize the path manager.
        
        Args:
            config_reader: ConfigReader instance for reading config values
            config_file_path: Optional path to the config file (for getting directory)
        """
        self.reader = config_reader
        self.config_file_path = config_file_path
        self._config_file_dir = None
        if config_file_path:
            self._config_file_dir = os.path.dirname(config_file_path)
        self.ensure_res_folder_exists()
        
    def ensure_res_folder_exists(self):
        """
        Ensure that the output results folder exists. Creates it if it doesn't exist.
        """
        res_folder = self.get_res_folder_name()
        if res_folder and not os.path.exists(res_folder):
            os.makedirs(res_folder)
    def get_data_folder_name(self) -> str:
        """
        Get the input data folder path from essentials section.
        
        Returns:
            str: Input folder path with trailing slash, or empty string if not found
        """
        if self.reader.has_option(essentials_field, in_folder_opt):
            folder_path = str(self.reader.get_option_of_field(essentials_field, in_folder_opt))
            return folder_path + '/' if folder_path else ''
        return ''
    
    def get_res_folder_name(self) -> str:
        """
        Get the output results folder path from essentials section.
        
        Returns:
            str: Output folder path with trailing slash, or empty string if not found
        """
        if self.reader.has_option(essentials_field, out_folder_opt):
            folder_path = str(self.reader.get_option_of_field(essentials_field, out_folder_opt))
            return folder_path + '/' if folder_path else ''
        return ''
    
    def get_prefix_name(self) -> str:
        """
        Get the output file prefix from essentials section.
        
        Returns:
            str: Output prefix name, or empty string if not found
        """
        if self.reader.has_option(essentials_field, out_prefix_opt):
            return str(self.reader.get_option_of_field(essentials_field, out_prefix_opt))
        return ''
    
    def get_config_file_dir(self) -> Optional[str]:
        """
        Get the directory containing the config file.
        
        Returns:
            str: Directory path, or None if config_file_path was not provided
        """
        return self._config_file_dir
    
    def get_config_file_path(self) -> Optional[str]:
        """
        Get the full path to the config file.
        
        Returns:
            str: Config file path, or None if not provided
        """
        return self.config_file_path
    
    def join_path(self, *paths: str) -> str:
        """
        Join path components using os.path.join.
        
        Args:
            *paths: Path components to join
            
        Returns:
            str: Joined path
        """
        return os.path.join(*paths)
    
    def ensure_trailing_slash(self, path: str) -> str:
        """
        Ensure a path has a trailing slash.
        
        Args:
            path: Path to process
            
        Returns:
            str: Path with trailing slash
        """
        if path and not path.endswith('/'):
            return path + '/'
        return path

