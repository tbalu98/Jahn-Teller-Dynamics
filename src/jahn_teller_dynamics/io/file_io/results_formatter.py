"""
Results formatting for Jahn-Teller calculations.

This module provides functions to format and structure theoretical results
and input data from Exe_tree objects for display and saving.
"""

from typing import Dict, List, Optional, TYPE_CHECKING, Any
import pandas as pd

if TYPE_CHECKING:
    import jahn_teller_dynamics.physics.quantum_physics as qmp


def format_theoretical_results_string(exe_tree: Any) -> str:
    """
    Format theoretical results as a human-readable string.
    
    Args:
        exe_tree: Exe_tree object containing theoretical results
        
    Returns:
        Formatted string with theoretical results
    """
    res_str = 'Theoretical results:\n'
    
    if exe_tree.p_factor is not None:
        res_str += '\n\tHam reduction factor = ' + str(round(abs(exe_tree.p_factor), 4))
    
    if exe_tree.lambda_theory is not None:
        res_str += '\n\tTheoretical energy level splitting = ' + str(round(abs(exe_tree.lambda_theory), 4)) + ' meV'
    
    # Add JT theory parameters if available
    # For Taylor coefficients (order_flag == 3), calculate E_JT and delta to show full format
    if exe_tree.JT_theory is not None:
        jt_theory = exe_tree.JT_theory
        
        # If using Taylor coefficients, calculate E_JT and delta to show full format
        if jt_theory.order_flag == 3:
            try:
                # Calculate E_JT and delta from F, G, and hw
                jt_theory.calc_parameters_from_Taylor_coeffs()
                # Set order_flag to 2 to get the full format in __repr__
                jt_theory.order_flag = 2
            except (ValueError, AttributeError):
                # If calculation fails, keep order_flag == 3 and use simple format
                pass
        
        # Append JT theory representation
        res_str += '\n-------------------------------------------------'
        res_str += '\n' + str(jt_theory)
    
    return res_str


def format_theoretical_results_dict(exe_tree: Any) -> Dict[str, List[str]]:
    """
    Format theoretical results as a dictionary for CSV export.
    
    Args:
        exe_tree: Exe_tree object containing theoretical results
        
    Returns:
        Dictionary with 'attribute' and 'values' keys, each containing lists
    """
    # Start with input data
    temp_res_dict = format_input_data_dict(exe_tree)
    
    # Add theoretical results
    if exe_tree.JT_theory is not None:
        jt_theory = exe_tree.JT_theory
        
        # If using Taylor coefficients (order_flag == 3), calculate E_JT and delta
        if jt_theory.order_flag == 3:
            try:
                # Calculate E_JT and delta from F, G, and hw
                jt_theory.calc_parameters_from_Taylor_coeffs()
            except (ValueError, AttributeError):
                # If calculation fails, E_JT and delta will not be available
                pass
        
        # Save E_JT if available (for order_flag 1, 2, or 3 after calculation)
        if hasattr(jt_theory, 'E_JT_meV') and jt_theory.E_JT_meV is not None:
            if jt_theory.order_flag == 1 or jt_theory.order_flag == 2 or jt_theory.order_flag == 3:
                temp_res_dict['Jahn-Teller energy (meV)'] = [jt_theory.E_JT_meV]
        
        # Save barrier energy if available (for order_flag 2, or 3 after calculation)
        if hasattr(jt_theory, 'delta_meV') and jt_theory.delta_meV is not None:
            if jt_theory.order_flag == 2 or jt_theory.order_flag == 3:
                temp_res_dict['barrier energy (meV)'] = [jt_theory.delta_meV]
        
        temp_res_dict['vibrational energy quantum (meV)'] = [jt_theory.hw_meV]
    
    temp_res_dict['Ham reduction factor'] = [abs(exe_tree.p_factor)] if exe_tree.p_factor is not None else [None]
    temp_res_dict['delta_p factor'] = [exe_tree.delta_p_factor] if exe_tree.delta_p_factor is not None else [None]
    temp_res_dict['delta_f factor'] = [exe_tree.delta_f_factor] if exe_tree.delta_f_factor is not None else [None]
    temp_res_dict['f factor'] = [exe_tree.f_factor] if exe_tree.f_factor is not None else [None]
    temp_res_dict['Energy splitting due to dynamic Jahn-Teller effect (meV)'] = [exe_tree.KJT_factor] if exe_tree.KJT_factor is not None else [None]
    temp_res_dict['Energy splitting due to spin-orbit coupling (meV)'] = [exe_tree.lambda_SOC] if exe_tree.lambda_SOC is not None else [None]
    temp_res_dict['Energy splitting (meV)'] = [abs(exe_tree.lambda_theory)] if exe_tree.lambda_theory is not None else [None]
    
    # Convert to attribute-value format
    res_dict = {
        'attribute': [str(x) for x in temp_res_dict.keys()],
        'values': [str(x[0]) if x[0] is not None else 'None' for x in temp_res_dict.values()]
    }
    
    return res_dict


def format_input_data_string(exe_tree: Any) -> str:
    """
    Format input data as a human-readable string.
    
    Args:
        exe_tree: Exe_tree object containing input data
        
    Returns:
        Formatted string with input data
    """
    res_str = 'Input data from ab initio calculations:\n'
    
    if exe_tree.JT_theory is not None:
        if exe_tree.JT_theory.symm_lattice is not None:
            res_str += '\tsymmetric geometry energy = ' + str(round(exe_tree.JT_theory.symm_lattice.energy, 4)) + ' eV\n'
        
        if exe_tree.JT_theory.JT_lattice is not None:
            res_str += '\tminimum geometry energy = ' + str(round(exe_tree.JT_theory.JT_lattice.energy, 4)) + ' eV\n'
        
        if exe_tree.JT_theory.order_flag == 2 and exe_tree.JT_theory.barrier_lattice is not None:
            res_str += '\tsaddle point geometry energy = ' + str(round(exe_tree.JT_theory.barrier_lattice.energy, 4)) + ' eV\n'
        
        if exe_tree.JT_theory.order_flag == 1 or exe_tree.JT_theory.order_flag == 2:
            res_str += '\tsymmetric - minimum geometry distance = ' + str(round(exe_tree.JT_theory.JT_dist, 4)) + ' Å √amu\n'
        
        if exe_tree.JT_theory.order_flag == 2:
            res_str += '\tsymmetric - saddle point geometry distance = ' + str(round(exe_tree.JT_theory.barrier_dist, 4)) + ' Å √amu\n'
    
    if exe_tree.intrinsic_soc is not None:
        res_str += '\tDFT spin-orbit coupling = ' + str(round(exe_tree.intrinsic_soc, 4)) + ' meV\n'
    
    if exe_tree.orbital_red_fact is not None:
        res_str += '\torbital reduction factor = ' + str(round(exe_tree.orbital_red_fact, 4)) + '\n'
    
    return res_str


def format_input_data_dict(exe_tree: Any) -> Dict[str, List]:
    """
    Format input data as a dictionary for CSV export.
    
    Args:
        exe_tree: Exe_tree object containing input data
        
    Returns:
        Dictionary with input data, where each value is a list (for consistency with results format)
    """
    res_dict = {}
    
    if exe_tree.JT_theory is not None:
        jt_theory = exe_tree.JT_theory
        
        # If using Taylor coefficients (order_flag == 3), calculate E_JT, delta, and distances
        if jt_theory.order_flag == 3:
            try:
                # Calculate E_JT, delta, JT_dist, and barrier_dist from F, G, and hw
                jt_theory.calc_parameters_from_Taylor_coeffs()
            except (ValueError, AttributeError):
                # If calculation fails, values will not be available
                pass
        
        if jt_theory.order_flag != 3:
            res_dict['symmetric geometry energy (eV)'] = [exe_tree.JT_theory.symm_lattice.energy] if exe_tree.JT_theory.symm_lattice is not None else [None]
            res_dict['minimum geometry energy (eV)'] = [exe_tree.JT_theory.JT_lattice.energy] if exe_tree.JT_theory.JT_lattice is not None else [None]
            res_dict['saddle point geometry energy (eV)'] = [exe_tree.JT_theory.barrier_lattice.energy] if (exe_tree.JT_theory.barrier_lattice is not None and exe_tree.JT_theory.order_flag == 2) else [None]
        
        # Save JT_dist and barrier_dist if available (for order_flag 1, 2, or 3 after calculation)
        if hasattr(exe_tree.JT_theory, 'JT_dist') and exe_tree.JT_theory.JT_dist is not None:
            if exe_tree.JT_theory.order_flag == 1 or exe_tree.JT_theory.order_flag == 2 or exe_tree.JT_theory.order_flag == 3:
                res_dict['high symmetry - minimum energy configuration distance (Å √amu)'] = [exe_tree.JT_theory.JT_dist]
        
        # Save barrier_dist (can be None when only F is present, G = 0)
        if hasattr(exe_tree.JT_theory, 'barrier_dist'):
            if exe_tree.JT_theory.order_flag == 2 or exe_tree.JT_theory.order_flag == 3:
                # Save barrier_dist even if it's None (for order_flag == 3 with only F)
                res_dict['high symmetry - saddle point configuration distance (Å √amu)'] = [exe_tree.JT_theory.barrier_dist]
    
    res_dict['DFT spin-orbit coupling (meV)'] = [exe_tree.intrinsic_soc] if exe_tree.intrinsic_soc is not None else [None]
    res_dict['orbital reduction factor'] = [exe_tree.orbital_red_fact] if exe_tree.orbital_red_fact is not None else [None]
    
    return res_dict


def save_theoretical_results(exe_tree: Any, res_path: str, separator: str = ';') -> None:
    """
    Save theoretical results to CSV file.
    
    Args:
        exe_tree: Exe_tree object containing theoretical results
        res_path: Path to save the CSV file
        separator: CSV separator (default: ';')
    """
    res_dict = format_theoretical_results_dict(exe_tree)
    res_df = pd.DataFrame(res_dict).set_index('attribute')
    res_df.to_csv(res_path, sep=separator)


def save_input_data(exe_tree: Any, res_path: str, calc_name: Optional[str] = None, separator: str = ';') -> None:
    """
    Save input data to CSV file.
    
    Args:
        exe_tree: Exe_tree object containing input data
        res_path: Path to save the CSV file
        calc_name: Optional calculation name to include in the data
        separator: CSV separator (default: ';')
    """
    input_data_res = format_input_data_dict(exe_tree)
    
    if calc_name is not None:
        input_data_res['calculation name'] = [calc_name]
    
    # Convert to attribute-value format
    res_dict = {
        'attribute': [str(x) for x in input_data_res.keys()],
        'values': [str(x[0]) if x[0] is not None else 'None' for x in input_data_res.values()]
    }
    
    res_df = pd.DataFrame(res_dict).set_index('attribute')
    res_df.to_csv(res_path, sep=separator)

