"""
JT_Calculator - High level helper for calculations on JT interaction objects.

This module provides a small facade class which operates on `Exe_tree`
objects (`JT_int`) that are created in the user workflow.  It also
offers a factory that builds such objects from a `JTConfigParser`,
thereby making use of the modern `JTTheoryBuilder` infrastructure.
"""

from typing import Any, Dict, List, Sequence, Tuple, Optional
import os

import pandas as pd
import jahn_teller_dynamics.physics.quantum_physics as qmp
import jahn_teller_dynamics.physics.jahn_teller_theory as jt
import jahn_teller_dynamics.math.maths as maths
import jahn_teller_dynamics.math.matrix_mechanics as mm
import jahn_teller_dynamics.io.file_io.csv_writer as csv_writer
from jahn_teller_dynamics.io.config.parser import JTConfigParser
from jahn_teller_dynamics.io.config.constants import mag_field_strength_csv_col


class JT_Calculator:
    """
    Helper class that performs calculations with `JT_int` (`Exe_tree`) objects.

    The class is intentionally lightweight: it can either be instantiated
    with an existing `JT_int` coming from the user workflow, or it can
    construct a new one from a `JTConfigParser` (which internally uses
    `JTTheoryBuilder`).
    """

    def __init__(self, JT_int: qmp.Exe_tree):
        """
        Initialize the calculator with an existing interaction object.

        Args:
            JT_int: `qmp.Exe_tree` instance representing the JT interaction.
        """
        self.JT_int = JT_int

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_config_parser(
        cls,
        JT_config_parser: JTConfigParser,
        section_to_look_for: str = "",
        complex_trf: bool = True,
    ) -> "JT_Calculator":
        """
        Build a `JT_int` from configuration and wrap it in a calculator.

        This builds a JT_int through the `JTConfigParser` facade
        which in turn delegates to `JTTheoryBuilder`.

        Args:
            JT_config_parser: Configuration parser instance.
            section_to_look_for: Name of the configuration section.
            complex_trf: Currently kept for backward compatibility.

        Returns:
            JT_Calculator: Calculator initialized with the created `JT_int`.
        """
        order = JT_config_parser.get_order()
        intrincis_soc = JT_config_parser.get_spin_orbit_coupling(section_to_look_for)
        orbital_red_fact = JT_config_parser.get_gL_factor(section_to_look_for)

        orientation_basis = JT_config_parser.get_system_orientation_basis()

        # Always build JT theory - it will determine if model Hamiltonian is needed
        JT_theory = JT_config_parser.create_Jahn_Teller_theory_from_cfg(
            section_to_look_for
        )

        # create_electron_phonon_Exe_tree will check JT_theory.order_flag
        # and create minimal_Exe_tree if order_flag == 0 (model Hamiltonian)
        JT_int = qmp.Exe_tree.create_electron_phonon_Exe_tree(
            JT_theory, order, intrincis_soc, orbital_red_fact, orientation_basis
        )

        # Only add spin system and interactions for non-model Hamiltonians
        if JT_theory.order_flag != 0 and intrincis_soc != 0.0:
            JT_int.add_spin_system()
            JT_int.create_one_mode_DJT_hamiltonian()
            JT_int.add_spin_orbit_coupling()

        return cls(JT_int)

    # ------------------------------------------------------------------
    # Magnetic interaction calculation
    # ------------------------------------------------------------------
    def _build_magnetic_hamiltonian(
        self,
        B_field: maths.col_vector,
        strain_fields: Optional[maths.col_vector] = None,
    ) -> qmp.Exe_tree:
        """
        Build and diagonalize the Hamiltonian with magnetic field (and optional strain) interaction.

        This method handles the transformation of the magnetic field into the appropriate
        basis and constructs the full DJT+SOC+magnetic (+strain) Hamiltonian.

        Args:
            B_field: Magnetic field vector (as `maths.col_vector`).
            strain_fields: Optional strain field vector.  If given, its
                interaction is added to the DJT+SOC+magnetic Hamiltonian.

        Returns:
            Exe_tree: The Hamiltonian with calculated eigen values and vectors.
        """
        JT_int = self.JT_int

        # Handle transformation into symmetry lattice basis if present
        if JT_int.JT_theory is not None and JT_int.JT_theory.symm_lattice is not None:
            B_field = B_field.in_new_basis(
                JT_int.JT_theory.symm_lattice.get_normalized_basis_vecs()
            )

        # Transform into the normalized basis of the interaction
        B_field = B_field.basis_trf(JT_int.get_normalized_basis_vecs())

        # Build Hamiltonian with magnetic interaction (and optional strain)
        H_DJT_mag = JT_int.create_DJT_SOC_mag_interaction(*B_field.tolist())
        if strain_fields is not None:
            H_DJT_mag = H_DJT_mag + JT_int.create_strain_field_interaction(
                strain_fields.tolist()
            )

        H_DJT_mag.calc_eigen_vals_vects()

        return H_DJT_mag

    def calc_magnetic_interaction(
        self,
        B_fields: Sequence[maths.col_vector],
        strain_fields: Optional[maths.col_vector] = None,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[Any]]]:
        """
        Calculate magnetic interaction energies and eigenkets.

        Calculate magnetic interaction energies and eigenkets
        for the internally stored `JT_int`.

        Args:
            B_fields: Iterable of magnetic field vectors (as `maths.col_vector`).
            strain_fields: Optional strain field vector.  If given, its
                interaction is added to the DJT+SOC+magnetic Hamiltonian.

        Returns:
            A tuple of two dictionaries:

            - energies_dict:  Mapping with keys
              ``'B_field', 'E0', 'E1', 'E2', 'E3'``.  Energies are stored
              in GHz (converted from meV).
            - eigen_kets_dict: Mapping with the same keys where
              eigenstates (kets) are stored instead of scalar energies.
        """
        JT_int = self.JT_int

        energy_labels = ["E0", "E1", "E2", "E3"]
        JT_int_Es_dict: Dict[str, List[Any]] = {
            "B_field": list(B_fields),
            "E0": [],
            "E1": [],
            "E2": [],
            "E3": [],
        }
        JT_int_eigen_kets_dict: Dict[str, List[Any]] = {
            "B_field": list(B_fields),
            "E0": [],
            "E1": [],
            "E2": [],
            "E3": [],
        }

        JT_int.create_one_mode_DJT_hamiltonian()

        for B_field in B_fields:
            H_DJT_mag = self._build_magnetic_hamiltonian(B_field, strain_fields)

            for eig_ket, line_label in zip(H_DJT_mag.eigen_kets, energy_labels):
                JT_int_Es_dict[line_label].append(
                    maths.meV_to_GHz(eig_ket.eigen_val)
                )
                JT_int_eigen_kets_dict[line_label].append(eig_ket)

        return JT_int_Es_dict, JT_int_eigen_kets_dict

    def calc_and_save_magnetic_interaction(
        self,
        B_fields: Sequence[maths.col_vector],
        B_field_strengths: Sequence[float],
        csv_writer: csv_writer.CSVWriter,
        res_folder: str,
        prefix_name: str,
        eigen_state_type: str,
        strain_fields: Optional[maths.col_vector] = None,
    ) -> Dict[str, List[Any]]:
        """
        Calculate magnetic interaction energies and save eigen values/vectors to files.

        This method calculates the magnetic field effect on the JT interaction for each
        B field value, saves the eigen vectors and values, and creates a summary CSV
        with energy states vs magnetic field strength.

        Args:
            B_fields: Iterable of magnetic field vectors (as `maths.col_vector`).
            B_field_strengths: Iterable of magnetic field strengths (in Tesla).
            csv_writer: CSVWriter instance for saving eigen vectors/values.
            res_folder: Results folder path where files will be saved.
            prefix_name: Prefix for output filenames.
            eigen_state_type: Type of eigen states ('real' or 'complex').
            strain_fields: Optional strain field vector.  If given, its
                interaction is added to the DJT+SOC+magnetic Hamiltonian.

        Returns:
            Dictionary with keys ``'B_field', 'E0', 'E1', 'E2', 'E3'``.  Energies are stored
            in GHz (converted from meV).  The structure matches ``calc_magnetic_interaction()``.
        """
        JT_int = self.JT_int
        energy_labels = ["E0", "E1", "E2", "E3"]
        JT_int_Es_dict: Dict[str, List[Any]] = {
            "B_field": list(B_fields),
            "E0": [],
            "E1": [],
            "E2": [],
            "E3": [],
        }

        # Ensure res_folder exists
        os.makedirs(res_folder, exist_ok=True)

        for B_field, B in zip(B_fields, B_field_strengths):
            # Build the magnetic Hamiltonian
            H_DJT_mag = self._build_magnetic_hamiltonian(B_field, strain_fields)
            
            # Set H_int so that calc_eigen_vals_vects() works correctly
            JT_int.H_int = H_DJT_mag

            fn_prefix = prefix_name + '_' + str(round(B, 4)) + 'T'

            # Save eigen vectors/values based on type
            if eigen_state_type == 'complex':
                # For complex, we need to transform the eigen vectors
                comp_eig_vecs = self.calc_and_transform_eigen_states()
                eigen_vec_path = os.path.join(res_folder, fn_prefix + '_complex_eigen_vectors.csv')
                eigen_val_path = os.path.join(res_folder, fn_prefix + '_complex_eigen_values.csv')
                csv_writer.write_eigen_vectors_and_values(
                    comp_eig_vecs, eigen_vec_path, eigen_val_path
                )
            elif eigen_state_type == 'real':
                # For real, use CSVWriter
                eig_vecs = JT_int.calc_eigen_vals_vects()
                eigen_vec_path = os.path.join(res_folder, fn_prefix + '_real_eigen_vectors.csv')
                eigen_val_path = os.path.join(res_folder, fn_prefix + '_real_eigen_values.csv')
                csv_writer.write_eigen_vectors_and_values(
                    eig_vecs, eigen_vec_path, eigen_val_path
                )
                comp_eig_vecs = eig_vecs
            else:
                raise ValueError(f"Unknown eigen_state_type: {eigen_state_type}. Must be 'real' or 'complex'.")

            # Collect energies
            for eig_ket, line_label in zip(comp_eig_vecs.eigen_kets, energy_labels):
                JT_int_Es_dict[line_label].append(maths.meV_to_GHz(eig_ket.eigen_val))

        # Save energy dependence CSV (use field strengths for CSV, but keep B_field in return dict)
        energy_dep_dict = JT_int_Es_dict.copy()
        energy_dep_dict[mag_field_strength_csv_col] = list(B_field_strengths)
        # Remove B_field from CSV since we're using field strengths instead
        energy_dep_dict.pop("B_field", None)
        
        # Rename energy columns from E0, E1, E2, E3 to eigenstate1 (GHz), eigenstate2 (GHz), etc.
        renamed_dict = {mag_field_strength_csv_col: energy_dep_dict[mag_field_strength_csv_col]}
        for i, old_label in enumerate(energy_labels, 1):
            if old_label in energy_dep_dict:
                renamed_dict[f'eigenstate_{i} (GHz)'] = energy_dep_dict[old_label]
        
        energy_dep_path = os.path.join(
            res_folder, prefix_name + '_magnetic_field_dependence_of_energy_states.csv'
        )
        df = pd.DataFrame(renamed_dict).set_index(mag_field_strength_csv_col)
        df.to_csv(energy_dep_path, sep=csv_writer.separator, index=csv_writer.index)
        
        return JT_int_Es_dict

    def calc_and_transform_eigen_states(self) -> Any:
        """
        Calculate and transform eigen vectors/values to complex representation.

        This method transforms the eigen vectors to the complex orbital basis.
        This is used when eigen_state_type is 'complex'. The saving should be
        done separately by the csv_writer.
        
        After transformation, the system_tree's base_states are updated to
        reflect the new basis (e.g., from ex,ey to e+,e-).

        Returns:
            eigen_vector_space: The transformed eigen vector space object.
        """
        JT_int = self.JT_int
        
        # Create basis transformation matrix from existing C_tr operator
        basis_trf_mx = JT_int.system_tree.create_operator(
            operator_id='C_tr', 
            operator_sys='orbital_system'
        )
        
        # Calculate eigen vectors/values
        eig_vecs = JT_int.calc_eigen_vals_vects()
        
        # Create new Hilbert space with e+/e- labels for the orbital system
        # Find orbital system node
        orbital_system = JT_int.system_tree.root_node.find_node('orbital_system')
        if orbital_system is None:
            raise ValueError("Orbital system not found in system tree")
        
        # Create new base_states with e+/e- labels
        new_orbital_basis = mm.hilber_space_bases().from_qm_nums_list(
            [['e+'], ['e-']],
            qm_nums_names=['orbital']
        )
        
        # Get all leaf systems and create new composite Hilbert space
        leaf_systems = JT_int.system_tree.root_node.find_leaves()
        new_leaf_bases = []
        for leaf in leaf_systems:
            if leaf.id == 'orbital_system':
                new_leaf_bases.append(new_orbital_basis)
            else:
                # Keep other leaf systems unchanged
                if leaf.base_states is not None:
                    new_leaf_bases.append(leaf.base_states)
        
        # Create new composite Hilbert space
        new_hilbert_space = mm.hilber_space_bases.kron_hilber_spaces(new_leaf_bases)
        
        # Transform to complex representation
        comp_eig_vecs = eig_vecs.transform_vector_space(
            new_hilbert_space,
            basis_trf_mx
        )
        
        # Update the orbital system's base_states to use e+/e- labels
        orbital_system.base_states = new_orbital_basis
        orbital_system.dim = new_orbital_basis.dim
        
        # Recreate root node's Hilbert space from updated children
        if JT_int.system_tree.root_node.has_child():
            JT_int.system_tree.root_node.create_hilbert_space()
        
        return comp_eig_vecs

    def calc_LzSz_expectation_values(
        self,
        LzSz_calc_num: int
    ) -> Dict[str, List[Any]]:
        """
        Calculate LzSz expectation values for eigenstates.
        
        This method calculates the LzSz operator expectation values for the
        first LzSz_calc_num eigenstates of the JT interaction.
        
        Args:
            LzSz_calc_num: Number of eigenstates to calculate LzSz for
            
        Returns:
            Dictionary with keys 'state_name', 'eigenenergy', 'LzSz' containing
            lists of state names, eigenenergies, and LzSz expectation values.
        """
        JT_int = self.JT_int
        
        LzSz_op = JT_int.system_tree.create_operator(
            'LzSz',
            subsys_id='point_defect',
            operator_sys='electron_system'
        )
        
        state_names = ['eigenstate_' + str(i) for i in range(0, LzSz_calc_num)]
        eigen_energies = [
            x.eigen_val for x in JT_int.H_int.eigen_kets[0:LzSz_calc_num]
        ]
        
        LzSz_expected_vals = [
            LzSz_op.calc_expected_val(eig_ket)
            for eig_ket in JT_int.H_int.eigen_kets[0:LzSz_calc_num]
        ]
        
        LzSz_res = {
            'state_name': state_names,
            'eigenenergy': eigen_energies,
            'LzSz': LzSz_expected_vals
        }
        
        return LzSz_res

    def no_soc_operation(self) -> None:
        """
        Perform no spin-orbit coupling operation on the JT interaction.
        
        This method calculates the reduction factor from first order perturbation
        theory for a system without spin-orbit coupling. It:
        1. Calculates eigen values and vectors
        2. Creates a degenerate system from the two ground states
        3. Sets up LzSz operator
        4. Adds perturbation and calculates reduction factor
        5. Stores the reduction factor in JT_int.p_factor
        """
        import jahn_teller_dynamics.math.matrix_mechanics as mm
        import jahn_teller_dynamics.physics.quantum_system as qs
        
        JT_int = self.JT_int
        eigen_kets = JT_int.calc_eigen_vals_vects()
        #JT_int.calc_reduction_factors()
        
        ground_1 = eigen_kets[0]
        ground_2 = eigen_kets[1]
        
        deg_sys = mm.degenerate_system_2D([ground_1, ground_2])
        
        electron_system = JT_int.system_tree.find_subsystem('electron_system')
        orbital_system = JT_int.system_tree.find_subsystem('orbital_system')
        
        spin_sys = qs.quantum_system_node.create_spin_system_node()
        
        Sz = spin_sys.operators['Sz']
        Lz = orbital_system.operators['Lz']
        
        electron_system.operators['LzSz'] = Lz ** Sz
        
        pert_ham_Lz = 0.5 * JT_int.system_tree.create_operator(
            operator_id='Lz',
            operator_sys='orbital_system'
        )
        
        print('Reduction factor from first order perturbation:')
        
        deg_sys.add_perturbation(pert_ham_Lz)
        
        Ham_red_fact = deg_sys.p_red_fact
        
        print('\n\t\tHam reduction factor = ' + str(round(Ham_red_fact, 4)))
        
        JT_int.p_factor = Ham_red_fact
        JT_int.lambda_Ham = None


# ==================== Standalone Functions ====================

def calc_transition_energies(
    ex_Es: Dict[str, List[float]],
    gnd_Es: Dict[str, List[float]],
    ex_labels: List[str],
    gnd_labels: List[str],
    field_strengths: List[float]
) -> Dict[str, List[float]]:
    """
    Calculate transition energies between excited and ground states.
    
    This function calculates the energy differences between excited state levels
    and ground state levels for each magnetic field strength, producing transition
    energies for different spectral lines.
    
    Args:
        ex_Es: Dictionary of excited state energies with keys like 'E0', 'E1', 'E2', 'E3'.
               Each value is a list of energies (in GHz) for different field strengths.
        gnd_Es: Dictionary of ground state energies with keys like 'E0', 'E1', 'E2', 'E3'.
                Each value is a list of energies (in GHz) for different field strengths.
        ex_labels: List of excited state labels to use (e.g., ['E0', 'E1'] or ['E2', 'E3']).
        gnd_labels: List of ground state labels to use (e.g., ['E0', 'E1'] or ['E2', 'E3']).
        field_strengths: List of magnetic field strengths (in Tesla).
        
    Returns:
        Dictionary with keys:
        - 'magnetic field (T)': List of field strengths
        - 'line_0 (GHz)': List of transition energies for first line
        - 'line_1 (GHz)': List of transition energies for second line
        - 'line_2 (GHz)': List of transition energies for third line
        - 'line_3 (GHz)': List of transition energies for fourth line
        
        The transitions are calculated as: ex_Es[ex_label][i] - gnd_Es[gnd_label][i]
        for each field strength i, iterating through all combinations of ex_labels
        and gnd_labels.
        
    Example:
        >>> ex_Es = {'E0': [100, 101], 'E1': [102, 103], 'E2': [104, 105], 'E3': [106, 107]}
        >>> gnd_Es = {'E0': [50, 51], 'E1': [52, 53], 'E2': [54, 55], 'E3': [56, 57]}
        >>> transitions = calc_transition_energies(
        ...     ex_Es, gnd_Es, ['E0', 'E1'], ['E0', 'E1'], [0.0, 1.0]
        ... )
        >>> transitions['line_0 (GHz)']
        [50.0, 50.0]  # E0(ex) - E0(gnd) for each field
    """
    line_labels = ['line_0 (GHz)', 'line_1 (GHz)', 'line_2 (GHz)', 'line_3 (GHz)']
    transitions = {
        'magnetic field (T)': field_strengths,
        'line_0 (GHz)': [],
        'line_1 (GHz)': [],
        'line_2 (GHz)': [],
        'line_3 (GHz)': []
    }
    
    for i in range(len(field_strengths)):
        line_label_iter = iter(line_labels)
        
        for j in range(len(ex_labels)):
            for k in range(len(gnd_labels)):
                line_label = next(line_label_iter)
                transitions[line_label].append(
                    ex_Es[ex_labels[j]][i] - gnd_Es[gnd_labels[k]][i]
                )
    
    return transitions
