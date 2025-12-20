"""
Quantum system tree construction.

This module provides factory functions to build quantum system trees
for different Jahn-Teller models.

System structures:
- Electron-phonon: point_defect -> [nuclei (mode_1), electron_system (orbital_system)]
- Spin-electron-phonon: point_defect -> [nuclei (mode_1), electron_system (orbital_system, spin_system)]
- Minimal model: point_defect -> [electron_system (orbital_system, spin_system)]
"""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import jahn_teller_dynamics.physics.quantum_system as qs
    import jahn_teller_dynamics.physics.jahn_teller_theory as jt
else:
    import jahn_teller_dynamics.physics.quantum_system as qs


def build_electron_phonon_system(
    jt_theory: 'jt.Jahn_Teller_Theory',
    order: int,
    spatial_dim: int = 2
) -> 'qs.quantum_system_tree':
    """
    Build quantum system tree for electron-phonon JT model.
    
    Structure:
        point_defect
        ├── nuclei
        │   └── mode_1 (phonon system)
        └── electron_system
            └── orbital_system
    
    Args:
        jt_theory: JT theory parameters (needs hw_meV attribute)
        order: Phonon truncation order
        spatial_dim: Spatial dimension (default: 2 for E⊗e JT)
        
    Returns:
        quantum_system_tree: Constructed system tree
    """
    from jahn_teller_dynamics.physics.quantum_physics import one_mode_phonon_sys
    
    # Build orbital system
    orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()
    electron_system = qs.quantum_system_node('electron_system', children=[orbital_system])
    
    # Build phonon system
    mode_1 = one_mode_phonon_sys(
        jt_theory.hw_meV, spatial_dim, order, 
        ['x', 'y'], 'mode_1', 'mode_1'
    )
    
    # Build nuclei
    nuclei = qs.quantum_system_node('nuclei')
    
    # Build point defect
    point_defect_node = qs.quantum_system_node(
        'point_defect', 
        children=[nuclei, electron_system]
    )
    
    # Create tree and insert phonon mode
    point_defect_tree = qs.quantum_system_tree(point_defect_node)
    point_defect_tree.insert_node('nuclei', mode_1)
    
    return point_defect_tree


def build_spin_electron_phonon_system(
    jt_theory: 'jt.Jahn_Teller_Theory',
    order: int,
    spatial_dim: int = 2
) -> 'qs.quantum_system_tree':
    """
    Build quantum system tree for spin-electron-phonon JT model.
    
    Structure:
        point_defect
        ├── nuclei
        │   └── mode_1 (phonon system)
        └── electron_system
            ├── orbital_system
            └── spin_system
    
    Args:
        jt_theory: JT theory parameters (needs hw_meV attribute)
        order: Phonon truncation order
        spatial_dim: Spatial dimension (default: 2 for E⊗e JT)
        
    Returns:
        quantum_system_tree: Constructed system tree
    """
    from jahn_teller_dynamics.physics.quantum_physics import one_mode_phonon_sys
    
    # Build orbital system
    orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()
    electron_system = qs.quantum_system_node('electron_system', children=[orbital_system])
    
    # Build spin system
    spin_sys = qs.quantum_system_node.create_spin_system_node()
    
    # Build phonon system
    mode_1 = one_mode_phonon_sys(
        jt_theory.hw_meV, spatial_dim, order, 
        ['x', 'y'], 'mode_1', 'mode_1'
    )
    
    # Build nuclei
    nuclei = qs.quantum_system_node('nuclei')
    
    # Build point defect
    point_defect_node = qs.quantum_system_node(
        'point_defect', 
        children=[nuclei, electron_system]
    )
    
    # Create tree and insert nodes
    point_defect_tree = qs.quantum_system_tree(point_defect_node)
    point_defect_tree.insert_node('nuclei', mode_1)
    point_defect_tree.insert_node('electron_system', spin_sys)
    
    return point_defect_tree


def build_minimal_model_system() -> 'qs.quantum_system_tree':
    """
    Build quantum system tree for minimal model (no phonons).
    
    Structure:
        point_defect
        └── electron_system
            ├── orbital_system
            └── spin_system
    
    This is used for the four-state model Hamiltonian where phonons
    are integrated out and only the electronic degrees of freedom remain.
    
    Returns:
        quantum_system_tree: Constructed system tree
    """
    # Build orbital and spin systems
    orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()
    spin_sys = qs.quantum_system_node.create_spin_system_node()
    
    # Build electron system
    electron_system = qs.quantum_system_node(
        'electron_system', 
        children=[orbital_system, spin_sys]
    )
    
    # Build point defect
    point_defect_system = qs.quantum_system_node(
        'point_defect', 
        children=[electron_system]
    )
    
    # Create tree
    return qs.quantum_system_tree(point_defect_system)

