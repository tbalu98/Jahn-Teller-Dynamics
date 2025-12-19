"""
ConfigReader - Low-level configuration file reading operations.

This class provides a clean interface for reading values from configuration files,
handling type conversions and missing values gracefully.
"""

from configparser import ConfigParser
from typing import Optional, List, Callable, Any


class ConfigReader:
    """
    Low-level reader for configuration files.
    
    This class wraps ConfigParser and provides convenient methods for reading
    configuration values with proper type handling and default values.
    """
    
    def __init__(self, config: ConfigParser):
        """
        Initialize the config reader with a ConfigParser instance.
        
        Args:
            config: A ConfigParser instance that has already read a config file
        """
        self.config = config
    
    def has_section(self, section: str) -> bool:
        """
        Check if a section exists in the config.
        
        Args:
            section: Section name to check
            
        Returns:
            bool: True if section exists, False otherwise
        """
        return self.config.has_section(section)
    
    def has_option(self, section: str, option: str) -> bool:
        """
        Check if an option exists in a section.
        
        Args:
            section: Section name
            option: Option name to check
            
        Returns:
            bool: True if option exists in section, False otherwise
        """
        return self.config.has_option(section, option)
    
    def get_option_of_field(self, section: str, option: str, default: str = '') -> str:
        """
        Get a string option from a section.
        
        Args:
            section: Section name
            option: Option name
            default: Default value if option doesn't exist (default: empty string)
            
        Returns:
            str: Option value as string, or default if not found
        """
        if self.config.has_option(section, option):
            return str(self.config[section][option])
        return default
    
    def get_float_option_of_field(self, section: str, option: str) -> Optional[float]:
        """
        Get a float option from a section.
        
        Args:
            section: Section name
            option: Option name
            
        Returns:
            float: Option value as float, or None if not found or invalid
        """
        if self.config.has_option(section, option):
            try:
                return float(str(self.config[section][option]))
            except (ValueError, TypeError):
                return None
        return None
    
    def get_int_option_of_field(self, section: str, option: str, default: int = 0) -> int:
        """
        Get an integer option from a section.
        
        Args:
            section: Section name
            option: Option name
            default: Default value if option doesn't exist (default: 0)
            
        Returns:
            int: Option value as integer, or default if not found or invalid
        """
        if self.config.has_option(section, option):
            try:
                value_str = str(self.config[section][option])
                return int(value_str) if value_str != '' else default
            except (ValueError, TypeError):
                return default
        return default
    
    def get_bool_option_of_field(self, section: str, option: str, default: bool = False) -> bool:
        """
        Get a boolean option from a section.
        
        Args:
            section: Section name
            option: Option name
            default: Default value if option doesn't exist (default: False)
            
        Returns:
            bool: Option value as boolean, or default if not found
        """
        if self.config.has_option(section, option):
            value = str(self.config[section][option]).lower().strip()
            return value == 'true'
        return default
    
    def get_splitted_strs(self, section: str, option: str, 
                         converter: Callable[[str], Any] = str) -> Optional[List[Any]]:
        """
        Get an option value and split it by commas, applying a converter function.
        
        This is useful for options like "1.0,2.0,3.0" that need to be parsed
        into a list of numbers.
        
        Args:
            section: Section name
            option: Option name
            converter: Function to apply to each split value (default: str)
            
        Returns:
            List[Any]: List of converted values, or None if option doesn't exist or is empty
        """
        value = self.get_option_of_field(section, option)
        if value == '':
            return None
        
        try:
            return [converter(item.strip()) for item in value.split(',') if item.strip()]
        except (ValueError, TypeError):
            return None
    
    def get_section(self, section: str) -> Optional[dict]:
        """
        Get all options from a section as a dictionary.
        
        Args:
            section: Section name
            
        Returns:
            dict: Dictionary of option: value pairs, or None if section doesn't exist
        """
        if not self.config.has_section(section):
            return None
        
        return dict(self.config[section])
    
    def get_all_sections(self) -> List[str]:
        """
        Get a list of all section names in the config.
        
        Returns:
            List[str]: List of section names
        """
        return self.config.sections()
    
    def get_raw_value(self, section: str, option: str) -> Optional[str]:
        """
        Get the raw string value from config without any conversion.
        
        Args:
            section: Section name
            option: Option name
            
        Returns:
            str: Raw option value, or None if not found
        """
        if self.config.has_option(section, option):
            return self.config[section][option]
        return None
    
    def conditional_option(self, section: str, option: str) -> bool:
        """
        Check if an option exists and is set to 'true'.
        
        This is a convenience method for boolean flags that may or may not exist.
        
        Args:
            section: Section name
            option: Option name
            
        Returns:
            bool: True if option exists and equals 'true', False otherwise
        """
        if not self.config.has_option(section, option):
            return False
        value = str(self.config[section][option]).lower().strip()
        return value == 'true'

