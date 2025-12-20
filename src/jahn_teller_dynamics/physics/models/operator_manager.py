"""
Operator management for quantum system trees.

This module provides functions to manage operator storage and retrieval patterns
used in Exe_tree, where operators are stored in subsystems and then referenced
at the root node level for Hamiltonian construction.
"""

from typing import TYPE_CHECKING, Optional
import jahn_teller_dynamics.physics.quantum_system as qs
import jahn_teller_dynamics.math.matrix_mechanics as mm

if TYPE_CHECKING:
    pass


def store_and_get_root_operator(
    system_tree: qs.quantum_system_tree,
    operator: mm.MatrixOperator,
    operator_id: str,
    subsystem_id: str,
    operator_sys: str,
    root_node_id: Optional[str] = None
) -> mm.MatrixOperator:
    """
    Store an operator in a subsystem and return root-level reference.
    
    This function implements the common pattern:
    1. Store operator in subsystem's operators dictionary
    2. Create root-level operator reference
    3. Return root-level operator for use in Hamiltonian
    
    Args:
        system_tree: The quantum system tree
        operator: The operator to store
        operator_id: Identifier for the operator (used for retrieval)
        subsystem_id: ID of the subsystem to store the operator in
        operator_sys: System type identifier (e.g., 'spin_system', 'orbital_system')
        root_node_id: Optional root node ID (if None, uses system_tree.root_node.id)
        
    Returns:
        Root-level operator reference that can be added to H_int
    """
    if root_node_id is None:
        root_node_id = system_tree.root_node.id
    
    # Store operator in subsystem
    subsystem = system_tree.find_subsystem(subsystem_id)
    subsystem.operators[operator_id] = operator
    
    # Create and return root-level operator reference
    root_operator = system_tree.create_operator(
        operator_id,
        subsys_id=root_node_id,
        operator_sys=operator_sys
    )
    
    return root_operator


def add_operator_to_hamiltonian(
    system_tree: qs.quantum_system_tree,
    operator: mm.MatrixOperator,
    operator_id: str,
    subsystem_id: str,
    operator_sys: str,
    coefficient: float = 1.0,
    root_node_id: Optional[str] = None
) -> mm.MatrixOperator:
    """
    Store operator in subsystem and return root-level operator with coefficient.
    
    This is a convenience function that combines storage and coefficient multiplication.
    
    Args:
        system_tree: The quantum system tree
        operator: The operator to store
        operator_id: Identifier for the operator
        subsystem_id: ID of the subsystem to store the operator in
        operator_sys: System type identifier
        coefficient: Coefficient to multiply the operator by (default: 1.0)
        root_node_id: Optional root node ID
        
    Returns:
        Root-level operator reference multiplied by coefficient
    """
    root_operator = store_and_get_root_operator(
        system_tree,
        operator,
        operator_id,
        subsystem_id,
        operator_sys,
        root_node_id
    )
    
    return coefficient * root_operator

