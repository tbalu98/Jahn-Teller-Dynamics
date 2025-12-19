"""
JTTheoryBuilder - Factory class for creating Jahn-Teller theory objects from configuration.

This class encapsulates all logic for building JT theory objects from various input sources:
- VASP XML files (vasprun.xml)
- CSV files with coordinate data
- Taylor coefficients (F, G, hw)
- Model Hamiltonian parameters
- JT parameters (E_JT, delta, hw, distances)
- Energy-distance pairs
"""

import jahn_teller_dynamics.physics.jahn_teller_theory as jt
import jahn_teller_dynamics.io.file_io.vasp as V
import jahn_teller_dynamics.io.file_io.xml_parser as xml_parser
import jahn_teller_dynamics.physics.quantum_physics as qmp
import jahn_teller_dynamics.math.maths as maths
from collections import namedtuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jahn_teller_dynamics.io.config.reader import ConfigReader
    from jahn_teller_dynamics.io.config.parameter_extractor import ParameterExtractor
    from jahn_teller_dynamics.io.utils.path_manager import PathManager
    from jahn_teller_dynamics.io.config.section_detector import SectionTypeDetector

# Import constants from the main config parsing module
from jahn_teller_dynamics.io.config.constants import (
    symm_latt_opt, JT_latt_opt, barr_latt_opt,
    atom_structure_field, symm_latt_energy_opt, min_energy_latt_energy_opt,
    saddle_point_latt_energy_opt, EJT_opt, E_barr_opt, hw_opt,
    symm_min_dist_opt, symm_saddl_dist_opt, F_opt, G_opt,
    Ham_red_opt, SOC_split_opt, f_factor_opt, delta_f_opt,
    Yx_opt, Yy_opt, K_JT_opt, con_int_en_opt, con_int_loc_opt,
    global_min_loc_opt, saddle_point_loc_opt, essentials_field,
    basis_vector_1_opt, basis_vector_2_opt, basis_vector_3_opt
)


class JTTheoryBuilder:
    """
    Factory class for building Jahn-Teller theory objects from configuration.
    
    This class handles the creation of JT theory objects from various input formats,
    delegating parameter extraction to other components.
    """
    
    def __init__(
        self,
        config_reader: 'ConfigReader',
        parameter_extractor: 'ParameterExtractor',
        path_manager: 'PathManager',
        type_detector: 'SectionTypeDetector'
    ):
        """
        Initialize the JT theory builder.
        
        Args:
            config_reader: Component for reading config file options
            parameter_extractor: Component for extracting physics parameters
            path_manager: Component for managing file paths
            type_detector: Component for detecting input source types
        """
        self.reader = config_reader
        self.params = parameter_extractor
        self.paths = path_manager
        self.detector = type_detector
    
    def create_Jahn_Teller_theory_from_cfg(self, section_to_look_for):
        """
        Main factory method that routes to appropriate builder based on input type.
        
        Args:
            section_to_look_for: Configuration section name to read from
            
        Returns:
            jt.Jahn_Teller_Theory: Constructed JT theory object
        """
        if self.detector.is_from_JT_pars(section_to_look_for):
            return self.build_JT_theories_from_cfg(section_to_look_for)
        elif self.detector.is_from_vasprun_xml(section_to_look_for):
            data_folder = self.paths.get_data_folder_name()
            return self.build_jt_theory_from_vasprunxmls(data_folder, section_to_look_for)
        elif self.detector.is_from_csv(section_to_look_for):
            return self.build_jt_theory_from_csv_and_pars_data(section_to_look_for)
        elif self.detector.is_from_Taylor_coeffs(section_to_look_for):
            return self.build_JT_theory_from_Taylor_coeffs(section_to_look_for)
        elif self.detector.is_from_model_Hamiltonian(section_to_look_for):
            return self.build_JT_theory_from_model_Hamiltonian_cfg(section_to_look_for)
        elif self.detector.is_from_energy_distance_pairs(section_to_look_for):
            return self.build_JT_theories_from_energy_distance_pairs(section_to_look_for)
        else:
            raise ValueError(f"Could not determine input type for section: {section_to_look_for}")
    
    def build_jt_theory_from_vasprunxmls(self, data_folder: str, section_to_look_for: str):
        """
        Build JT theory from VASP XML files (vasprun.xml).
        
        Args:
            data_folder: Path to folder containing XML files
            section_to_look_for: Configuration section name
            
        Returns:
            jt.Jahn_Teller_Theory: Constructed JT theory object
        """
        symm_latt_fn = data_folder + self.reader.get_option_of_field(section_to_look_for, symm_latt_opt)
        JT_geom_fn = data_folder + self.reader.get_option_of_field(section_to_look_for, JT_latt_opt)
        barr_geom_fn = None
        if self.reader.has_option(section_to_look_for, barr_latt_opt):
            barr_geom_fn = data_folder + self.reader.get_option_of_field(section_to_look_for, barr_latt_opt)
        
        symm_geom = xml_parser.xml_parser(symm_latt_fn).lattice
        JT_geom = xml_parser.xml_parser(JT_geom_fn).lattice
        barr_geom = xml_parser.xml_parser(barr_geom_fn).lattice if barr_geom_fn is not None else None
        
        return jt.Jahn_Teller_Theory(symm_geom, JT_geom, barr_geom)
    
    def build_jt_theory_from_csv_and_pars_data(self, section_to_look_for: str):
        """
        Build JT theory from CSV files with coordinate data.
        
        Args:
            section_to_look_for: Configuration section name
            
        Returns:
            jt.Jahn_Teller_Theory: Constructed JT theory object
        """
        data_folder = self.paths.get_data_folder_name()
        symm_latt_csv_dir = data_folder + self.reader.get_option_of_field(section_to_look_for, symm_latt_opt)
        JT_latt_csv_dir = data_folder + self.reader.get_option_of_field(section_to_look_for, JT_latt_opt)
        barr_latt_csv_fn = self.reader.get_option_of_field(section_to_look_for, barr_latt_opt)
        barr_latt_csv_dir = data_folder + barr_latt_csv_fn if barr_latt_csv_fn else None
        
        # Get basis vectors and atom data
        basis_vecs = self._get_basis_vectors(atom_structure_field)
        if basis_vecs is None:
            raise ValueError(f"Basis vectors not found in section: {atom_structure_field}")
        
        atom_masses = self.params.get_masses(atom_structure_field)
        atom_numbers = self.params.get_numbers(atom_structure_field)
        atom_names = self.params.get_atom_names(atom_structure_field)
        
        # Create atom data structures
        atom_data = namedtuple('atom_data', 'name mass number')
        atom_datas = [
            atom_data(name, mass, number)
            for name, mass, number in zip(atom_names, atom_masses, atom_numbers)
        ]
        
        # Get lattice energies
        sym_lattice_energy = self.params.get_lattice_energy(section_to_look_for, symm_latt_energy_opt)
        less_symm_lattice_1_energy = self.params.get_lattice_energy(section_to_look_for, min_energy_latt_energy_opt)
        
        # Build lattices
        symm_lattice = V.Lattice().read_from_coordinates_dataframe(
            symm_latt_csv_dir, atom_datas, basis_vecs, sym_lattice_energy
        )
        less_symm_lattice_1 = V.Lattice().read_from_coordinates_dataframe(
            JT_latt_csv_dir, atom_datas, basis_vecs, less_symm_lattice_1_energy
        )
        
        if barr_latt_csv_fn:
            less_symm_lattice_2_energy = self.params.get_lattice_energy(
                section_to_look_for, saddle_point_latt_energy_opt
            )
            less_symm_lattice_2 = V.Lattice().read_from_coordinates_dataframe(
                barr_latt_csv_dir, atom_datas, basis_vecs, less_symm_lattice_2_energy
            )
        else:
            less_symm_lattice_2 = None
        
        return jt.Jahn_Teller_Theory(symm_lattice, less_symm_lattice_1, less_symm_lattice_2)
    
    def build_JT_theory_from_Taylor_coeffs(self, section_to_look_for: str):
        """
        Build JT theory from Taylor coefficients (F, G, hw).
        
        Args:
            section_to_look_for: Configuration section name
            
        Returns:
            jt.Jahn_Teller_Theory: Constructed JT theory object
        """
        F = self.params.get_F_coeff(section_to_look_for)
        G = self.params.get_G_coeff(section_to_look_for)
        hw = self.params.get_hw(section_to_look_for)
        return jt.Jahn_Teller_Theory().from_Taylor_coeffs(hw, F, G)
    
    def build_JT_theory_from_model_Hamiltonian_cfg(self, section_to_look_for: str):
        """
        Build JT theory from model Hamiltonian parameters.
        
        Args:
            section_to_look_for: Configuration section name
            
        Returns:
            jt.Jahn_Teller_Theory: Constructed JT theory object
        """
        lambda_DFT = self.params.get_SOC_split(section_to_look_for)
        KJT = self.params.get_KJT(section_to_look_for)
        #if KJT is None:
        #    raise ValueError(f"K_JT parameter not found in section: {section_to_look_for}")
        gL = self.params.get_gL_factor(section_to_look_for)
        delta_f = self.params.get_delta_f(section_to_look_for)
        Yx = self.params.get_Yx(section_to_look_for)
        Yy = self.params.get_Yy(section_to_look_for)
        f_factor = self.params.get_f_factor(section_to_look_for)
        
        return jt.Jahn_Teller_Theory().from_minimal_model_parameters(
            lambda_DFT,  gL, delta_f, Yx, Yy, f_factor
        )
    
    def build_JT_theories_from_cfg(self, state_JT_field: str):
        """
        Build JT theory from JT parameters (E_JT, delta, hw, distances).
        
        Args:
            state_JT_field: Configuration section name
            
        Returns:
            jt.Jahn_Teller_Theory: Constructed JT theory object
        """
        if not self.reader.has_section(state_JT_field):
            raise ValueError(f"Section {state_JT_field} not found in config")
        
        E_JT_str = self.reader.get_option_of_field(state_JT_field, EJT_opt)
        if not E_JT_str:
            raise ValueError(f"E_JT parameter not found in section: {state_JT_field}")
        E_JT = float(E_JT_str)
        
        delta_field = self.reader.get_option_of_field(state_JT_field, E_barr_opt)
        delta_meV = float(delta_field) if delta_field else 0.0
        
        hw_str = self.reader.get_option_of_field(state_JT_field, hw_opt)
        if not hw_str:
            raise ValueError(f"hw parameter not found in section: {state_JT_field}")
        hw = float(hw_str)
        
        JT_theory = jt.Jahn_Teller_Theory()
        JT_theory.E_JT_meV = E_JT
        JT_theory.delta_meV = delta_meV
        JT_theory.hw_meV = hw
        JT_theory.order_flag = 2
        
        JT_dist_str = self.reader.get_option_of_field(state_JT_field, symm_min_dist_opt)
        barr_dist_str = self.reader.get_option_of_field(state_JT_field, symm_saddl_dist_opt)
        if not JT_dist_str or not barr_dist_str:
            raise ValueError(f"Distance parameters not found in section: {state_JT_field}")
        
        JT_theory.JT_dist = float(JT_dist_str)
        JT_theory.barrier_dist = float(barr_dist_str)
        
        JT_theory.calc_parameters_until_second_order_from_JT_pars()
        
        JT_theory.orbital_red_fact = self.params.get_gL_factor(state_JT_field)
        JT_theory.intrinsic_soc = self.params.get_spin_orbit_coupling(state_JT_field)
        
        return JT_theory
    
    def build_JT_theories_from_energy_distance_pairs(self, field_to_look_for: str):
        """
        Build JT theory from energy-distance pairs.
        
        Args:
            field_to_look_for: Configuration section name
            
        Returns:
            jt.Jahn_Teller_Theory: Constructed JT theory object
        """
        con_int_en = self.reader.get_float_option_of_field(field_to_look_for, con_int_en_opt)
        con_int_loc = self.reader.get_float_option_of_field(field_to_look_for, con_int_loc_opt)
        
        saddle_point_en = self.reader.get_float_option_of_field(field_to_look_for, saddle_point_latt_energy_opt)
        saddle_point_loc = self.reader.get_float_option_of_field(field_to_look_for, saddle_point_loc_opt)
        
        minimum_en = self.reader.get_float_option_of_field(field_to_look_for, min_energy_latt_energy_opt)
        minimum_loc = self.reader.get_float_option_of_field(field_to_look_for, global_min_loc_opt)
        
        if any(x is None for x in [con_int_en, con_int_loc, saddle_point_en, 
                                    saddle_point_loc, minimum_en, minimum_loc]):
            raise ValueError(f"Missing required energy-distance parameters in section: {field_to_look_for}")
        
        JT_en = abs(con_int_en - minimum_en)
        barr_en = abs(saddle_point_en - minimum_en)
        
        JT_dist = abs(con_int_loc - minimum_loc)
        barr_dist = abs(con_int_loc - saddle_point_loc)
        
        JT_theory = jt.Jahn_Teller_Theory()
        JT_theory.E_JT_meV = JT_en
        JT_theory.JT_dist = JT_dist
        JT_theory.delta_meV = barr_en
        JT_theory.barrier_dist = barr_dist
        JT_theory.order_flag = 2
        
        JT_theory.calc_hw()
        JT_theory.calc_Taylor_coeffs_K()
        
        JT_theory.orbital_red_fact = self.params.get_gL_factor(field_to_look_for)
        JT_theory.intrinsic_soc = self.params.get_spin_orbit_coupling(field_to_look_for)
        
        return JT_theory
    
    def create_minimal_Exe_tree_from_cfg(self, section_to_look_for: str):
        """
        Create a minimal Exe_tree from configuration.
        
        Args:
            section_to_look_for: Configuration section name
            
        Returns:
            qmp.minimal_Exe_tree: Constructed minimal Exe tree
        """
        energy_split = self.params.get_SOC_split(section_to_look_for)
        if energy_split is None:
            raise ValueError(f"SOC_split parameter not found in section: {section_to_look_for}")
        
        gL = self.params.get_gL_factor(section_to_look_for)
        f_factor = self.params.get_f_factor(section_to_look_for)
        delta_f = self.params.get_delta_f(section_to_look_for)
        Yx = self.params.get_Yx(section_to_look_for)
        Yy = self.params.get_Yy(section_to_look_for)
        
        # Get orientation basis from essentials field
        orientation_basis = self._get_basis_col_vectors(essentials_field)
        if orientation_basis is None:
            raise ValueError(f"Orientation basis vectors not found in section: {essentials_field}")
        
        return qmp.minimal_Exe_tree.from_cfg_data(
            energy_split, orientation_basis, gL, delta_f, f_factor, Yx, Yy
        )
    
    # Helper methods (these should ideally be in FieldVectorParser, but kept here for now)
    def _get_basis_vectors(self, field_name: str):
        """
        Get basis vectors for a field (helper method).
        
        Note: This should ideally be in FieldVectorParser component.
        
        Args:
            field_name: Configuration section name
            
        Returns:
            List[V.Vector]: List of three basis vectors, or None if not found
        """
        b1_vec = self._get_basis_vector(field_name, basis_vector_1_opt)
        b2_vec = self._get_basis_vector(field_name, basis_vector_2_opt)
        b3_vec = self._get_basis_vector(field_name, basis_vector_3_opt)
        if b1_vec is None or b2_vec is None or b3_vec is None:
            return None
        return [b1_vec, b2_vec, b3_vec]
    
    def _get_basis_vector(self, section_id: str, option_id: str):
        """
        Get a single basis vector (helper method).
        
        Note: This should ideally be in FieldVectorParser component.
        
        Args:
            section_id: Configuration section name
            option_id: Option name for the basis vector
            
        Returns:
            V.Vector: Basis vector, or None if not found
        """
        res = self.reader.get_option_of_field(section_id, option_id)
        if res == '':
            return None
        coordinates = self.reader.get_splitted_strs(section_id, option_id, float)
        if coordinates is None or len(coordinates) != 3:
            return None
        return V.Vector(*coordinates)
    
    def _get_basis_col_vectors(self, field_name: str):
        """
        Get basis column vectors (helper method).
        
        Note: This should ideally be in FieldVectorParser component.
        
        Args:
            field_name: Configuration section name
            
        Returns:
            List[maths.col_vector]: List of three normalized column vectors, or None if not found
        """
        b1 = self._get_col_vector(field_name, basis_vector_1_opt)
        b2 = self._get_col_vector(field_name, basis_vector_2_opt)
        b3 = self._get_col_vector(field_name, basis_vector_3_opt)
        if b1 is None or b2 is None or b3 is None:
            return None
        return [b1.normalize(), b2.normalize(), b3.normalize()]
    
    def _get_col_vector(self, field_name: str, opt_name: str):
        """
        Get a column vector (helper method).
        
        Note: This should ideally be in FieldVectorParser component.
        
        Args:
            field_name: Configuration section name
            opt_name: Option name for the vector
            
        Returns:
            maths.col_vector: Column vector, or None if not found
        """
        coordinates = self.reader.get_splitted_strs(field_name, opt_name, float)
        if coordinates is None:
            return None
        return maths.col_vector.from_list(coordinates)

