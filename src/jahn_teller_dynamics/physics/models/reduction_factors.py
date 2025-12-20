"""
Reduction factors computation for Jahn-Teller systems.

This module provides functions to compute reduction factors and related quantities
used in Jahn-Teller theory, such as p_factor, f_factor, and K_JT_factor.
"""

from typing import NamedTuple
import jahn_teller_dynamics.physics.quantum_system as qs
import jahn_teller_dynamics.math.matrix_mechanics as mm


class ReductionFactorsResult(NamedTuple):
    """Result of reduction factors computation."""
    p_32: float
    p_12: float
    p_factor: float
    lambda_Ham: float
    delta_p_factor: float
    f_factor: float
    delta_f_factor: float
    lambda_SOC: float
    lambda_theory: float


def compute_reduction_factors(
    system_tree: qs.quantum_system_tree,
    H_int: mm.MatrixOperator,
    orbital_red_fact: float,
    intrinsic_soc: float
) -> ReductionFactorsResult:
    """
    Compute reduction factors for Jahn-Teller system.
    
    This function calculates the p and f reduction factors based on the
    expectation values of LzSz operator in the eigenstates of H_int.
    
    Args:
        system_tree: The quantum system tree containing the system structure
        H_int: The interaction Hamiltonian with computed eigenstates
        orbital_red_fact: Orbital reduction factor (g_L)
        intrinsic_soc: Intrinsic spin-orbit coupling strength
        
    Returns:
        ReductionFactorsResult containing all computed reduction factors
        
    Raises:
        ValueError: If H_int is None or eigenstates are not computed
        ValueError: If fewer than 3 eigenstates are available
    """
    if H_int is None:
        raise ValueError("H_int must be calculated before computing reduction factors")
    if not hasattr(H_int, 'eigen_kets') or H_int.eigen_kets is None:
        raise ValueError("Eigenvalues must be calculated before computing reduction factors")
    if len(H_int.eigen_kets) < 3:
        raise ValueError("Need at least 3 eigenstates to compute reduction factors")
    
    root_node_id = system_tree.root_node.id
    
    # Create LzSz operator at root node level
    LzSz_op = system_tree.create_operator(
        'LzSz',
        subsys_id=root_node_id,
        operator_sys='electron_system'
    )
    
    # Get eigenstates (index 2 is E_3/2, index 0 is E_1/2)
    E_32 = H_int.eigen_kets[2]
    E_12 = H_int.eigen_kets[0]
    
    # Calculate p factors
    p_32 = 2 * LzSz_op.calc_expected_val(E_32)
    p_12 = -2 * LzSz_op.calc_expected_val(E_12)
    
    p_factor = (p_32 + p_12) / 2
    lambda_Ham = p_factor
    
    delta_p_factor = (p_32 - p_12) / 2
    
    # Calculate f factors
    f_factor = -orbital_red_fact * p_factor
    delta_f_factor = orbital_red_fact * delta_p_factor
    
    # Calculate spin-orbit coupling
    lambda_SOC = p_factor * intrinsic_soc
    
    # Calculate theoretical lambda (energy splitting)
    lambda_theory = H_int.eigen_kets[2].eigen_val - H_int.eigen_kets[0].eigen_val.real
    
    return ReductionFactorsResult(
        p_32=p_32,
        p_12=p_12,
        p_factor=p_factor,
        lambda_Ham=lambda_Ham,
        delta_p_factor=delta_p_factor,
        f_factor=f_factor,
        delta_f_factor=delta_f_factor,
        lambda_SOC=lambda_SOC,
        lambda_theory=lambda_theory
    )


def compute_K_JT_factor(
    system_tree: qs.quantum_system_tree,
    H_int: mm.MatrixOperator
) -> float:
    """
    Compute K_JT factor from Jahn-Teller interaction.
    
    The K_JT factor is the difference in expectation values of H_DJT
    between the E_3/2 and E_1/2 eigenstates.
    
    Args:
        system_tree: The quantum system tree containing H_DJT operator
        H_int: The interaction Hamiltonian with computed eigenstates
        
    Returns:
        KJT_factor: The computed K_JT factor
        
    Raises:
        ValueError: If H_int is None or eigenstates are not computed
        ValueError: If fewer than 3 eigenstates are available
        KeyError: If H_DJT operator is not found in root node operators
    """
    if H_int is None:
        raise ValueError("H_int must be calculated before computing K_JT_factor")
    if not hasattr(H_int, 'eigen_kets') or H_int.eigen_kets is None:
        raise ValueError("Eigenvalues must be calculated before computing K_JT_factor")
    if len(H_int.eigen_kets) < 3:
        raise ValueError("Need at least 3 eigenstates to compute K_JT_factor")
    
    # Get H_DJT operator from root node
    if 'H_DJT' not in system_tree.root_node.operators:
        raise KeyError("H_DJT operator not found in root node operators. "
                      "Ensure create_one_mode_DJT_hamiltonian() or "
                      "create_multi_mode_hamiltonian() has been called first.")
    
    H_DJT = system_tree.root_node.operators['H_DJT']
    
    # Get eigenstates (index 2 is E_3/2, index 0 is E_1/2)
    E_32 = H_int.eigen_kets[2]
    E_12 = H_int.eigen_kets[0]
    
    # Calculate expectation values
    K_JT_32 = H_DJT.calc_expected_val(E_32)
    K_JT_12 = H_DJT.calc_expected_val(E_12)
    
    KJT_factor = K_JT_32 - K_JT_12
    
    return KJT_factor

