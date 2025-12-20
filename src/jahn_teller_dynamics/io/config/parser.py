"""
JTConfigParser - Main configuration file parser that orchestrates all components.

This class serves as the main entry point for reading and parsing Jahn-Teller
configuration files. It initializes and coordinates all component classes:
- ConfigReader: Low-level config reading
- ParameterExtractor: Physics parameter extraction
- PathManager: File path management
- SectionTypeDetector: Input type detection
- JTTheoryBuilder: JT theory object creation
"""

from configparser import ConfigParser
from typing import Optional, TYPE_CHECKING
import os

from jahn_teller_dynamics.io.config.reader import ConfigReader
from jahn_teller_dynamics.io.config.parameter_extractor import ParameterExtractor
from jahn_teller_dynamics.io.utils.path_manager import PathManager
from jahn_teller_dynamics.io.config.section_detector import SectionTypeDetector
from jahn_teller_dynamics.io.theory.builder import JTTheoryBuilder
from jahn_teller_dynamics.io.config.writer import ConfigWriter
from jahn_teller_dynamics.io.config.field_parser import FieldVectorParser
from jahn_teller_dynamics.io.file_io.csv_writer import CSVWriter
from jahn_teller_dynamics.io.visualization.plotter import Plotter

if TYPE_CHECKING:
    pass  # Type hints are handled inline


class JTConfigParser:
    """
    Main configuration file parser for Jahn-Teller calculations.
    
    This class reads a configuration file and provides access to all parsing
    functionality through composed component classes. It serves as a facade
    that coordinates all the specialized components.
    """
    
    def __init__(self, config_file_path: str):
        """
        Initialize the config parser by reading a configuration file.
        
        Args:
            config_file_path: Path to the .cfg configuration file
        """
        # Read config file
        with open(config_file_path, 'r') as config_file:
            config_string = config_file.read()
        
        self.config_file_path = config_file_path
        self.config_file_dir = os.path.dirname(config_file_path)
        
        # Create ConfigParser
        self.config = ConfigParser()
        self.config.read_string(config_string)
        
        # Initialize components
        self.reader = ConfigReader(self.config)
        self.params = ParameterExtractor(self.reader)
        self.paths = PathManager(self.reader, config_file_path)
        self.detector = SectionTypeDetector(self.reader)
        self.fields = FieldVectorParser(self.reader)
        self.builder = JTTheoryBuilder(self.reader, self.params, self.paths, self.detector)
        self.writer = ConfigWriter(self.reader, self.params, self.paths, self.config, self.config_file_dir)
        self.csv_writer = CSVWriter(path_manager=self.paths)
        self.plotter = Plotter(path_manager=self.paths)
        
        # Initialize default values (for backward compatibility)
        self._init_default_values()
    
    def _init_default_values(self):
        """Initialize default values as instance attributes."""
        self.input_folder = self.paths.get_data_folder_name()
        self.output_folder = self.paths.get_res_folder_name()
        self.max_vib_quant = self.params.get_order()
        self.output_prefix_name = self.paths.get_prefix_name()
        self.save_raw_parameters = self.detector.is_save_raw_pars()
        self.spectrum_range = self.params.get_calc_LzSz()
        self.eigen_states_type = self.params.get_eigen_state_type()
    
    # ==================== Delegate Methods for Backward Compatibility ====================
    
    # Config reading methods
    def get_option_of_field(self, field: str, option: str) -> str:
        """Get a string option from a section."""
        return self.reader.get_option_of_field(field, option)
    
    def get_float_option_of_field(self, field: str, option: str) -> Optional[float]:
        """Get a float option from a section."""
        return self.reader.get_float_option_of_field(field, option)
    
    def has_option(self, field: str, option: str) -> bool:
        """Check if an option exists in a section."""
        return self.reader.has_option(field, option)
    
    def has_section(self, section: str) -> bool:
        """Check if a section exists."""
        return self.reader.has_section(section)
    
    # Parameter extraction methods
    def get_spin_orbit_coupling(self, section: str) -> float:
        """Get spin-orbit coupling value."""
        return self.params.get_spin_orbit_coupling(section)
    
    def get_gL_factor(self, section: str = None) -> float:
        """Get orbital reduction factor (gL factor)."""
        if section is None:
            from jahn_teller_dynamics.io.config.constants import so_c_field
            return self.params.get_gL_factor(so_c_field)
        return self.params.get_gL_factor(section)
    
    def get_order(self) -> int:
        """Get maximum number of vibrational quanta."""
        return self.params.get_order()
    
    def get_F_coeff(self, section: str) -> float:
        """Get F coefficient."""
        return self.params.get_F_coeff(section)
    
    def get_G_coeff(self, section: str) -> float:
        """Get G coefficient."""
        return self.params.get_G_coeff(section)
    
    def get_hw(self, section: str) -> float:
        """Get vibrational energy quantum."""
        return self.params.get_hw(section)
    
    def get_f_factor(self, section: str) -> Optional[float]:
        """Get f factor."""
        return self.params.get_f_factor(section)
    
    def get_delta_f(self, section: str) -> float:
        """Get delta_f parameter."""
        return self.params.get_delta_f(section)
    
    def get_Yx(self, section: str) -> Optional[float]:
        """Get Yx parameter."""
        return self.params.get_Yx(section)
    
    def get_Yy(self, section: str) -> Optional[float]:
        """Get Yy parameter."""
        return self.params.get_Yy(section)
    
    def get_KJT(self, section: str) -> Optional[float]:
        """Get K_JT parameter."""
        return self.params.get_KJT(section)
    
    def get_SOC_split(self, section: str) -> Optional[float]:
        """Get SOC split parameter."""
        return self.params.get_SOC_split(section)
    
    def get_numbers(self, section: str):
        """Get list of atom numbers."""
        return self.params.get_numbers(section)
    
    def get_atom_names(self, section: str):
        """Get list of atom names."""
        return self.params.get_atom_names(section)
    
    def get_masses(self, section: str):
        """Get list of atom masses."""
        return self.params.get_masses(section)
    
    def get_lattice_energy(self, section: str, energy_opt: str) -> float:
        """Get lattice energy."""
        return self.params.get_lattice_energy(section, energy_opt)
    
    def get_calc_LzSz(self) -> int:
        """Get number of LzSz calculations."""
        return self.params.get_calc_LzSz()
    
    def get_eigen_state_type(self) -> str:
        """Get eigenstate type (real or complex)."""
        return self.params.get_eigen_state_type()
    
    # Path management methods
    def get_data_folder_name(self) -> str:
        """Get input data folder path."""
        return self.paths.get_data_folder_name()
    
    def get_res_folder_name(self) -> str:
        """Get output results folder path."""
        return self.paths.get_res_folder_name()
    
    def get_prefix_name(self) -> str:
        """Get output file prefix."""
        return self.paths.get_prefix_name()
    
    # Type detection methods
    def is_from_JT_pars(self, section: str) -> bool:
        """Check if section has JT parameters."""
        return self.detector.is_from_JT_pars(section)
    
    def is_from_vasprun_xml(self, section: str) -> bool:
        """Check if section specifies XML files."""
        return self.detector.is_from_vasprun_xml(section)
    
    def is_from_csv(self, section: str) -> bool:
        """Check if section specifies CSV files."""
        return self.detector.is_from_csv(section)
    
    def is_from_Taylor_coeffs(self, section: str) -> bool:
        """Check if section has Taylor coefficients."""
        return self.detector.is_from_Taylor_coeffs(section)
    
    def is_from_model_Hamiltonian(self, section: str) -> bool:
        """Check if section has model Hamiltonian parameters."""
        return self.detector.is_from_model_Hamiltonian(section)
    
    def is_from_energy_distance_pairs(self, section: str) -> bool:
        """Check if section has energy-distance pairs."""
        return self.detector.is_from_energy_distance_pairs(section)
    
    def is_ZPL_calculation(self) -> bool:
        """Check if this is a ZPL calculation."""
        return self.detector.is_ZPL_calculation()
    
    def is_single_case(self) -> bool:
        """Check if this is a single case calculation."""
        return self.detector.is_single_case()
    
    def is_save_raw_pars(self) -> bool:
        """Check if raw parameters should be saved."""
        return self.detector.is_save_raw_pars()
    
    def is_save_model_Hamiltonian_cfg(self) -> bool:
        """Check if model Hamiltonian config should be saved."""
        return self.detector.is_save_model_Hamiltonian_cfg()
    
    def is_save_Taylor_coeffs_cfg(self) -> bool:
        """Check if Taylor coefficients config should be saved."""
        return self.detector.is_save_Taylor_coeffs_cfg()
    
    def is_use_model_hamiltonian(self) -> bool:
        """Check if model Hamiltonian should be used."""
        return self.detector.is_use_model_hamiltonian()
    
    def is_real_eigen_vects(self) -> bool:
        """Check if real eigenstates are requested."""
        return self.detector.is_real_eigen_vects()
    
    def is_complex_eigen_vects(self) -> bool:
        """Check if complex eigenstates are requested."""
        return self.detector.is_complex_eigen_vects()
    
    # JT Theory building methods
    def create_Jahn_Teller_theory_from_cfg(self, section: str):
        """Create a Jahn-Teller theory object from configuration."""
        return self.builder.create_Jahn_Teller_theory_from_cfg(section)
    
    def create_minimal_Exe_tree_from_cfg(self, section: str):
        """Create a minimal Exe_tree from configuration."""
        return self.builder.create_minimal_Exe_tree_from_cfg(section)
    
    # ==================== Direct Component Access ====================
    
    @property
    def config_reader(self) -> ConfigReader:
        """Get the ConfigReader component."""
        return self.reader
    
    @property
    def parameter_extractor(self) -> ParameterExtractor:
        """Get the ParameterExtractor component."""
        return self.params
    
    @property
    def path_manager(self) -> PathManager:
        """Get the PathManager component."""
        return self.paths
    
    @property
    def section_type_detector(self) -> SectionTypeDetector:
        """Get the SectionTypeDetector component."""
        return self.detector
    
    @property
    def jt_theory_builder(self) -> JTTheoryBuilder:
        """Get the JTTheoryBuilder component."""
        return self.builder
    
    @property
    def config_writer(self) -> 'ConfigWriter':
        """Get the ConfigWriter component."""
        return self.writer
    
    @property
    def csv_writer_component(self) -> CSVWriter:
        """Get the CSVWriter component."""
        return self.csv_writer
    
    @property
    def plotter_component(self) -> Plotter:
        """Get the Plotter component."""
        return self.plotter
    
    @property
    def field_vector_parser(self) -> FieldVectorParser:
        """Get the FieldVectorParser component."""
        return self.fields
    
    # ==================== Field Vector Methods ====================
    
    def get_magnetic_field_vectors(self):
        """Get list of magnetic field vectors."""
        return self.fields.get_magnetic_field_vectors()
    
    def get_field_vectors(self, field_name: str):
        """Get list of field vectors for a given field name."""
        return self.fields.get_field_vectors(field_name)
    
    def get_strain_field_vector(self):
        """Get strain field vector."""
        return self.fields.get_strain_field_vector()
    
    def get_mag_dir_vector(self):
        """Get magnetic field direction vector."""
        return self.fields.get_mag_dir_vector()
    
    def get_strain_dir_vector(self):
        """Get strain field direction vector."""
        return self.fields.get_strain_dir_vector()
    
    def get_basis_col_vectors(self, field_name: str):
        """Get basis column vectors for a field."""
        return self.fields.get_basis_col_vectors(field_name)
    
    def get_system_orientation_basis(self):
        """Get system orientation basis vectors."""
        return self.fields.get_system_orientation_basis()
    
    def get_mag_field_strengths_list(self):
        """Get list of magnetic field strengths."""
        return self.fields.get_mag_field_strengths_list()
    
    def get_B_min(self) -> float:
        """Get minimum magnetic field strength."""
        return self.fields.get_B_min()
    
    def get_B_max(self) -> float:
        """Get maximum magnetic field strength."""
        return self.fields.get_B_max()
    
    def get_step_num(self) -> int:
        """Get number of steps for magnetic field."""
        return self.fields.get_step_num()
    
    # ==================== Config Writing Methods ====================
    
    def save_raw_pars(self, JT_int):
        """Save raw parameters for a single case calculation."""
        return self.writer.save_raw_pars(JT_int)
    
    def save_model_pars(self, JT_int):
        """Save model Hamiltonian parameters for a single case calculation."""
        return self.writer.save_model_pars(JT_int)
    
    def save_raw_pars_Taylor(self, JT_int):
        """Save Taylor coefficients config for a single case calculation."""
        return self.writer.save_raw_pars_Taylor(JT_int)
    
    def save_raw_pars_ZPL(self, JT_int_gnd, JT_int_ex):
        """Save raw parameters for a ZPL calculation."""
        return self.writer.save_raw_pars_ZPL(JT_int_gnd, JT_int_ex)
    
    def save_raw_pars_ZPL_model(self, JT_int_gnd, JT_int_ex):
        """Save model Hamiltonian config for ZPL calculation."""
        return self.writer.save_raw_pars_ZPL_model(JT_int_gnd, JT_int_ex)
    
    def save_raw_pars_ZPL_Taylor(self, JT_int_gnd, JT_int_ex):
        """Save Taylor coefficients config for ZPL calculation."""
        return self.writer.save_raw_pars_ZPL_Taylor(JT_int_gnd, JT_int_ex)
    
    def save_all_ZPL_outputs(self, JT_int_gnd, JT_int_ex):
        """
        Save all requested ZPL calculation outputs based on configuration.
        
        This method checks the configuration flags and saves:
        - Raw parameters (if save_raw_pars is enabled)
        - Model Hamiltonian config (if save_model_Hamiltonian_cfg is enabled)
        - Taylor coefficients config (if save_Taylor_coeffs_cfg is enabled)
        
        Args:
            JT_int_gnd: Exe_tree instance for ground state
            JT_int_ex: Exe_tree instance for excited state
        """
        if self.is_save_raw_pars():
            self.save_raw_pars_ZPL(JT_int_gnd, JT_int_ex)
        
        if self.is_save_model_Hamiltonian_cfg():
            self.save_raw_pars_ZPL_model(JT_int_gnd, JT_int_ex)
        
        if self.is_save_Taylor_coeffs_cfg():
            self.save_raw_pars_ZPL_Taylor(JT_int_gnd, JT_int_ex)

