"""
JTOrchestrator - High-level orchestrator for Jahn-Teller calculations and IO operations.

This class integrates all functionalities:
- Configuration parsing
- Jahn-Teller theory creation
- Calculation execution
- Results writing (CSV, plots)
- Workflow management (ZPL, single case)
"""

from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import os

import jahn_teller_dynamics.physics.quantum_physics as qmp
import jahn_teller_dynamics.io.theory.calculator as JT_Calculator
from jahn_teller_dynamics.io.config.parser import JTConfigParser
from jahn_teller_dynamics.io.file_io.csv_writer import CSVWriter
from jahn_teller_dynamics.io.visualization.plotter import Plotter
# results_formatter is imported inside functions that need it to avoid circular dependencies
from jahn_teller_dynamics.io.theory.calculator import calc_transition_energies
from jahn_teller_dynamics.io.utils import create_directory
from jahn_teller_dynamics.io.config.constants import single_case_section
import jahn_teller_dynamics.math.maths as maths


class JTOrchestrator:
    """
    Main orchestrator for Jahn-Teller calculations and IO operations.
    
    This class coordinates all aspects of a Jahn-Teller calculation:
    - Configuration parsing
    - Theory object creation
    - Calculation execution
    - Results saving and plotting
    """
    
    def __init__(self, config_parser: JTConfigParser):
        """
        Initialize the orchestrator with a configuration parser.
        
        Args:
            config_parser: JTConfigParser instance with loaded configuration
        """
        self.config = config_parser
        self.csv_writer = config_parser.csv_writer_component
        self.plotter = config_parser.plotter_component
        
        # Store commonly used components
        self.paths = config_parser.paths
        self.params = config_parser.params
        
    def run(self) -> Any:
        """
        Main entry point - determines calculation type and executes appropriate workflow.
        
        Returns:
            Results of the calculation (varies by workflow type)
        """
        # Check calculation type using the config parser
        if self.config.is_ZPL_calculation():
            return self.run_zpl_calculation()
        elif self.config.is_single_case():
            return self.run_single_case_calculation(single_case_section)
        else:
            raise ValueError("Could not determine calculation type from config file.")
    
    def run_zpl_calculation(self) -> Dict[str, Any]:
        """
        Execute Zero Phonon Line (ZPL) calculation workflow.
        
        This involves:
        1. Processing ground state
        2. Processing excited state
        3. Calculating transition energies
        4. Plotting and saving results
        
        Returns:
            Dictionary containing calculation results
        """
        calculation_name = self.config.get_prefix_name()
        
        print('-------------------------------------------------')
        print('Ground state:')
        ground_state = self._process_state('ground_state_parameters')
        
        print('-------------------------------------------------')
        print('Excited state:')
        excited_state = self._process_state('excited_state_parameters')
        
        # Save all requested ZPL outputs
        self.config.save_all_ZPL_outputs(ground_state['JT_int'], excited_state['JT_int'])
        
        # Calculate magnetic interactions
        B_fields = self.config.get_magnetic_field_vectors()
        strain_fields = self.config.get_strain_field_vector()
        
        if B_fields is not None:
            print('-------------------------------------------------')
            print('Calculating magnetic field effect...')
            B_field_strengths = self.config.get_mag_field_strengths_list()
            print(f'  Magnetic field range: {B_field_strengths[0]:.2f} - {B_field_strengths[-1]:.2f} T ({len(B_field_strengths)} steps)')
            if strain_fields is not None:
                print(f'  Strain field: {strain_fields.tolist()}')
            
            print('  Calculating ground state magnetic field dependence...')
            gnd_energies, _ = ground_state['calculator'].calc_magnetic_interaction(
                B_fields, strain_fields
            )
            print('  Calculating excited state magnetic field dependence...')
            ex_energies, _ = excited_state['calculator'].calc_magnetic_interaction(
                B_fields, strain_fields
            )
            print('  Completed magnetic field calculations')
        else:
            gnd_energies = None
            ex_energies = None
        
        # Calculate transition energies (only if magnetic field is present)
        if B_fields is not None:
            Bs = self.config.get_mag_field_strengths_list()
            transitions = self._calculate_all_transitions(ex_energies, gnd_energies, Bs)
            
            # Plot results
            self._plot_zpl_transitions(transitions, Bs, calculation_name)
            
            # Save transition data
            self._save_transition_data(transitions)
        else:
            transitions = None
        
        return {
            'ground_state': ground_state,
            'excited_state': excited_state,
            'transitions': transitions,
            'field_strengths': Bs
        }
    
    def run_single_case_calculation(self, section: str) -> Dict[str, Any]:
        """
        Execute single case calculation workflow.
        
        Args:
            section: Configuration section name to process
            
        Returns:
            Dictionary containing calculation results
        """
        result = self._process_state(section)
        
        # Handle magnetic field calculations if requested
        B_fields = self.config.get_magnetic_field_vectors()
        if B_fields is not None and not self.config.is_ZPL_calculation():
            print('-------------------------------------------------')
            print('Calculating magnetic field effect...')
            B_field_strengths = self.config.get_mag_field_strengths_list()
            print(f'  Magnetic field range: {B_field_strengths[0]:.2f} - {B_field_strengths[-1]:.2f} T ({len(B_field_strengths)} steps)')
            
            if self.config.is_use_model_hamiltonian():
                result['JT_int'] = qmp.minimal_Exe_tree.from_Exe_tree(result['JT_int'])
            
            strain_fields = self.config.get_strain_field_vector()
            if strain_fields is not None:
                print(f'  Strain field: {strain_fields.tolist()}')
            
            energies_dict = result['calculator'].calc_and_save_magnetic_interaction(
                B_fields,
                B_field_strengths,
                self.csv_writer,
                self.config.get_res_folder_name(),
                self.config.get_prefix_name(),
                self.config.get_eigen_state_type(),
                strain_fields
            )
            result['magnetic_interaction'] = energies_dict
            print('  Completed magnetic field calculation')
        
        return result
    
    def _process_state(self, section: str) -> Dict[str, Any]:
        """
        Process a single state (ground or excited) through the full workflow.
        
        Args:
            section: Configuration section name
            
        Returns:
            Dictionary containing JT_int, calculator, and other state information
        """
        # Create JT interaction using the calculator
        calculator = JT_Calculator.JT_Calculator.from_config_parser(
            self.config, section_to_look_for=section
        )
        JT_int = calculator.JT_int
        
        # Get configuration parameters
        order = self.config.get_order()
        intrinsic_soc = self.config.get_spin_orbit_coupling(section)
        
        print('-------------------------------------------------')
        print(f'Maximum number of energy quantums of vibrations in each direction = {order}')
        
        # Process based on spin-orbit coupling
        if intrinsic_soc != 0.0:
            self._process_with_soc(JT_int, section, order, calculator)
        elif self.config.is_from_model_Hamiltonian(section) and intrinsic_soc == 0.0:
            # Model Hamiltonian without SOC - minimal processing
            pass
        else:
            self._process_without_soc(JT_int, section, calculator)
        
        # Save raw parameters and model parameters if requested
        self._save_parameters_if_requested(JT_int)
        
        # Print theoretical results only if not already printed by _process_with_soc
        # (which prints via format_theoretical_results_string)
        # For non-SOC cases (both regular and model Hamiltonian), print here
        if intrinsic_soc == 0.0:
            # For _process_without_soc and model Hamiltonian cases, print theoretical results
            if JT_int.JT_theory is not None:
                # Ensure Taylor coefficients are formatted correctly
                if JT_int.JT_theory.order_flag == 3:
                    try:
                        JT_int.JT_theory.calc_parameters_from_Taylor_coeffs()
                        JT_int.JT_theory.order_flag = 2
                    except (ValueError, AttributeError):
                        pass
                print('-------------------------------------------------')
                print(JT_int.JT_theory)
        
        return {
            'JT_int': JT_int,
            'calculator': calculator,
            'section': section
        }
    
    def _process_with_soc(
        self, 
        JT_int: qmp.Exe_tree, 
        section: str, 
        order: int,
        calculator: JT_Calculator.JT_Calculator
    ) -> None:
        """
        Process state with spin-orbit coupling.
        
        Args:
            JT_int: Exe_tree object
            section: Configuration section name
            order: Order parameter
            calculator: JT_Calculator instance for this state
        """
        # Add spin system and calculate
        JT_int.add_spin_system()
        JT_int.create_one_mode_DJT_hamiltonian()
        JT_int.add_spin_orbit_coupling()
        JT_int.calc_eigen_vals_vects()
        
        # Calculate reduction factors
        JT_int.calc_reduction_factors()
        JT_int.calc_K_JT_factor()
        
        print('-------------------------------------------------')
        # Lazy import to avoid circular dependencies - import directly from module
        from jahn_teller_dynamics.io.file_io.results_formatter import format_theoretical_results_string
        print(format_theoretical_results_string(JT_int))
        
        # Save theoretical results
        th_res_name = section + '_theoretical_results.csv'
        self.csv_writer.write_theoretical_results_to_output(JT_int, th_res_name)
        
        # Save eigen vectors/values based on type
        eigen_state_type = self.config.get_eigen_state_type()
        calc_name = self.config.get_prefix_name()
        res_folder = self.config.get_res_folder_name()
        
        if eigen_state_type == 'real':
            self.csv_writer.write_eigen_vectors_and_values_to_output(
                JT_int.calc_eigen_vals_vects(),
                eigen_vec_suffix=section + '_eigen_vectors.csv',
                eigen_val_suffix=section + '_eigen_values.csv'
            )
        elif eigen_state_type == 'complex':
            comp_eig_vecs = calculator.calc_and_transform_eigen_states()
            self.csv_writer.write_eigen_vectors_and_values_to_output(
                comp_eig_vecs,
                eigen_vec_suffix=section + '_eigen_vectors.csv',
                eigen_val_suffix=section + '_eigen_values.csv'
            )
        
        # Calculate LzSz if requested
        LzSz_calc_num = self.config.get_calc_LzSz()
        if LzSz_calc_num > 0:
            self.LzSz_process(calculator, LzSz_calc_num, calc_name)
    
    def _process_without_soc(
        self, 
        JT_int: qmp.Exe_tree, 
        section: str,
        calculator: JT_Calculator.JT_Calculator
    ) -> None:
        """
        Process state without spin-orbit coupling.
        
        Args:
            JT_int: Exe_tree object
            section: Configuration section name
            calculator: JT_Calculator instance for this state
        """
        JT_int.create_one_mode_DJT_hamiltonian()
        
        # Perform no SOC operation using calculator
        calculator.no_soc_operation()
        
        create_directory(self.config.get_res_folder_name())
        
        th_res_name = 'theoretical_results.csv'
        self.csv_writer.write_theoretical_results_to_output(JT_int, th_res_name)
        self.csv_writer.write_eigen_vectors_and_values_to_output(
            JT_int.calc_eigen_vals_vects(),
            eigen_vec_suffix=section + '_eigen_vectors.csv',
            eigen_val_suffix=section + '_eigen_values.csv'
        )
    
    def _save_parameters_if_requested(self, JT_int: qmp.Exe_tree) -> None:
        """
        Save raw parameters and model parameters if requested in the config file.
        
        This method checks the configuration flags and saves:
        - Raw parameters (if save_raw_parameters is true)
        - Model Hamiltonian parameters (if save_model_hamiltonian_cfg is true)
        - Taylor coefficients config (if save_taylor_coeffs_cfg is true)
        
        Args:
            JT_int: Exe_tree instance containing calculation results
        """
        if self.config.is_save_raw_pars():
            self.config.save_raw_pars(JT_int)
        
        if self.config.is_save_model_Hamiltonian_cfg():
            self.config.save_model_pars(JT_int)
        
        if self.config.is_save_Taylor_coeffs_cfg():
            self.config.save_raw_pars_Taylor(JT_int)
    
    def LzSz_process(
        self,
        calculator: JT_Calculator.JT_Calculator,
        LzSz_calc_num: int,
        calc_name: str
    ) -> None:
        """
        Process LzSz expectation value calculation, plotting, and saving.
        
        This method assembles the LzSz workflow:
        1. Calculate LzSz expectation values using the calculator
        2. Plot the results using the plotter
        3. Create DataFrame and save to CSV using the csv_writer
        
        Args:
            calculator: JT_Calculator instance with initialized JT_int
            LzSz_calc_num: Number of eigenstates to calculate LzSz for
            calc_name: Calculation name/prefix for output files
        """
        # Calculate LzSz expectation values
        LzSz_data = calculator.calc_LzSz_expectation_values(LzSz_calc_num)
        
        # Plot LzSz expectation values
        res_folder = self.config.get_res_folder_name()
        plot_path = os.path.join(res_folder, calc_name + '_LzSz_plot.png')
        self.plotter.plot_LzSz_expectation_values(
            LzSz_data['LzSz'],
            LzSz_data['eigenenergy'],
            save_path=plot_path
        )
        
        # Create DataFrame and save to CSV
        LzSz_res_df = self.csv_writer.create_LzSz_dataframe(LzSz_data)
        self.csv_writer.write_LzSz_expected_values_to_output(
            LzSz_res_df, calc_name + '_LzSz_expected_values.csv'
        )
    
    def _calculate_all_transitions(
        self,
        ex_energies: Dict[str, List[float]],
        gnd_energies: Dict[str, List[float]],
        field_strengths: List[float]
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Calculate all transition energies (A, B, C, D).
        
        Args:
            ex_energies: Excited state energies dictionary
            gnd_energies: Ground state energies dictionary
            field_strengths: List of magnetic field strengths
            
        Returns:
            Dictionary with keys 'A', 'B', 'C', 'D' containing transition data
        """
        return {
            'A': calc_transition_energies(ex_energies, gnd_energies, ['E2', 'E3'], ['E0', 'E1'], field_strengths),
            'B': calc_transition_energies(ex_energies, gnd_energies, ['E2', 'E3'], ['E2', 'E3'], field_strengths),
            'C': calc_transition_energies(ex_energies, gnd_energies, ['E0', 'E1'], ['E0', 'E1'], field_strengths),
            'D': calc_transition_energies(ex_energies, gnd_energies, ['E0', 'E1'], ['E2', 'E3'], field_strengths)
        }
    
    def _plot_zpl_transitions(
        self,
        transitions: Dict[str, Dict[str, List[float]]],
        field_strengths: List[float],
        calculation_name: str
    ) -> None:
        """
        Plot ZPL transition results.
        
        Args:
            transitions: Dictionary of transition data
            field_strengths: List of magnetic field strengths
            calculation_name: Name for the calculation
        """
        results_folder = self.config.get_res_folder_name()
        fig_fn = calculation_name + "_ZPL_calculation.png"
        save_path = results_folder + fig_fn
        
        self.plotter.plot_ZPL_transitions(
            A_transition=transitions['A'],
            B_transition=transitions['B'],
            C_transition=transitions['C'],
            D_transition=transitions['D'],
            field_strengths=field_strengths,
            calculation_name=calculation_name,
            save_path=save_path,
            figsize=(14, 10),
            dpi=700,
            B_min=self.config.get_B_min(),
            B_max=self.config.get_B_max()
        )
    
    def _save_transition_data(self, transitions: Dict[str, Dict[str, List[float]]]) -> None:
        """
        Save transition energy data to CSV files.
        
        Args:
            transitions: Dictionary of transition data
        """
        self.csv_writer.write_transitions_to_output(transitions['A'], 'A_transitions.csv')
        self.csv_writer.write_transitions_to_output(transitions['B'], 'B_transitions.csv')
        self.csv_writer.write_transitions_to_output(transitions['C'], 'C_transitions.csv')
        self.csv_writer.write_transitions_to_output(transitions['D'], 'D_transitions.csv')

