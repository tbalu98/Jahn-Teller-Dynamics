"""
Observables computation for Jahn-Teller systems.

This module provides functions to compute physical observables such as
transition intensities and magnetic field-dependent eigenstates.
"""

from typing import List, Tuple, Optional, Dict, Any, Callable
import jahn_teller_dynamics.physics.quantum_system as qs
import jahn_teller_dynamics.math.matrix_mechanics as mm
import jahn_teller_dynamics.math.maths as maths


def compute_transition_intensities(
    system_tree: qs.quantum_system_tree,
    from_kets: List[mm.ket_vector],
    to_kets: List[mm.ket_vector],
    include_z: bool = False
) -> Tuple[List[float], List[float]]:
    """
    Compute transition intensities between eigenstates.
    
    Calculates the transition intensities using dipole operators:
        I = |<from|X|to>|² + |<from|Y|to>|² [+ |<from|Z|to>|²]
    
    Args:
        system_tree: The quantum system tree for creating operators
        from_kets: List of initial state ket vectors
        to_kets: List of final state ket vectors
        include_z: If True, include Z component in intensity calculation
        
    Returns:
        Tuple of (transition_intensities, transition_energies):
        - transition_intensities: List of transition intensities
        - transition_energies: List of energy differences (from - to)
    """
    # Create orbital operators
    x_op = system_tree.create_operator('X_orb')
    y_op = system_tree.create_operator('Y_orb')
    z_op = system_tree.create_operator('Z_orb')
    
    transition_intensities = []
    transition_energies = []
    
    # Convert kets to bras for expectation value calculations
    from_bras = [ket.to_bra_vector() for ket in from_kets]
    
    for from_bra in from_bras:
        for to_ket in to_kets:
            # Calculate transition intensity: |<from|X|to>|² + |<from|Y|to>|²
            x_int = abs((from_bra * x_op) * to_ket) ** 2
            y_int = abs((from_bra * y_op) * to_ket) ** 2
            
            transition_int = x_int + y_int
            
            # Optionally include Z component
            if include_z:
                z_int = abs((from_bra * z_op) * to_ket) ** 2
                transition_int += z_int
            
            transition_intensities.append(transition_int)
            transition_energies.append(from_bra.eigen_val - to_ket.eigen_val)
    
    return transition_intensities, transition_energies


def compute_magnetic_interaction_eigen_kets(
    system_tree: qs.quantum_system_tree,
    B_fields: List[maths.col_vector],
    create_hamiltonian_func: Callable[[float, float, float], mm.MatrixOperator],
    create_strain_func: Optional[Callable[[List[float]], mm.MatrixOperator]] = None,
    strain_fields: Optional[maths.col_vector] = None,
    normalized_basis_vecs: Optional[List[maths.col_vector]] = None,
    symm_lattice_basis_vecs: Optional[List[maths.col_vector]] = None,
    num_states: int = 4
) -> Dict[str, List[Any]]:
    """
    Compute eigenstates for different magnetic field configurations.
    
    This function calculates eigenstates for a series of magnetic field vectors,
    optionally including strain field interactions.
    
    Args:
        system_tree: The quantum system tree
        B_fields: List of magnetic field vectors
        create_hamiltonian_func: Function that creates the Hamiltonian given (Bx, By, Bz)
        create_strain_func: Optional function to create strain field interaction
        strain_fields: Optional strain field vector
        normalized_basis_vecs: Optional normalized basis vectors for field transformation
        symm_lattice_basis_vecs: Optional symmetry lattice basis vectors for field transformation
        num_states: Number of eigenstates to return (default: 4 for E0, E1, E2, E3)
        
    Returns:
        Dictionary with keys:
        - 'B_field': List of magnetic field vectors
        - 'E0', 'E1', 'E2', 'E3': Lists of eigenstate ket vectors for each field
    """
    # Initialize result dictionary
    res_dict = {'B_field': B_fields}
    for i in range(num_states):
        res_dict[f'E{i}'] = []
    
    for B_field in B_fields:
        # Transform B_field to symmetry lattice basis if provided
        if symm_lattice_basis_vecs is not None:
            B_field = B_field.in_new_basis(symm_lattice_basis_vecs)
        
        # Transform to normalized basis if provided
        if normalized_basis_vecs is not None:
            B_field = B_field.basis_trf(normalized_basis_vecs)
        
        # Create Hamiltonian with magnetic field
        H_DJT_mag = create_hamiltonian_func(*B_field.tolist())
        
        # Add strain field interaction if provided
        if strain_fields is not None and create_strain_func is not None:
            H_DJT_mag = H_DJT_mag + create_strain_func(strain_fields.tolist())
        
        # Calculate eigenvalues and eigenvectors
        H_DJT_mag.calc_eigen_vals_vects(
            quantum_states_bases=system_tree.root_node.base_states
        )
        
        # Store eigenstates
        for i in range(min(num_states, len(H_DJT_mag.eigen_kets))):
            res_dict[f'E{i}'].append(H_DJT_mag.eigen_kets[i])
    
    return res_dict

