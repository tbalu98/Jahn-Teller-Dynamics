import jahn_teller_dynamics.math.matrix_mechanics as mm
import jahn_teller_dynamics.math.braket_formalism as bf
from jahn_teller_dynamics.math.matrix_mechanics import MatrixOperator
import numpy as np
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import jahn_teller_dynamics.physics.jahn_teller_theory as jt
else:
    import jahn_teller_dynamics.physics.jahn_teller_theory as jt
import jahn_teller_dynamics.math.maths as maths
import jahn_teller_dynamics.physics.quantum_system as qs
import copy
import pandas as pd

# Import new hamiltonian modules
from jahn_teller_dynamics.physics.hamiltonians import (
    djt_hamiltonian,
    spin_orbit,
    field_interactions,
)

# Import system builder, reduction factors, and operator manager modules
from jahn_teller_dynamics.physics.models import (
    system_builder,
    compute_reduction_factors,
    compute_K_JT_factor,
    add_operator_to_hamiltonian,
    store_and_get_root_operator,
)

# Import numerics modules
from jahn_teller_dynamics.physics.numerics import (
    compute_transition_intensities,
    compute_magnetic_interaction_eigen_kets,
)

# Import constants
from jahn_teller_dynamics.physics.constants import (
    hbar_meVs,
    Bohn_magneton_meV_T,
    g_factor,
    round_precision_dig,
)

class one_mode_phonon_sys(qs.quantum_system_node):

    def create_complex_basis_gen_op(self):

        raise_x_op = bf.raise_index_operator(0)

        raise_y_op = bf.raise_index_operator(1)

        raise_x_mx_op = self.mx_op_builder.create_MatrixOperator(raise_x_op).truncate_matrix(self.trunc_num)
        
        raise_y_mx_op = self.mx_op_builder.create_MatrixOperator(raise_y_op).truncate_matrix(self.trunc_num)




        plus_gen_op = (1/mm.SQRT_2)*(raise_x_mx_op+complex(0.0,1.0)*raise_y_mx_op)
        minus_gen_op = (1/mm.SQRT_2)*(raise_x_mx_op+complex(0.0,-1.0)*raise_y_mx_op)

        return [plus_gen_op, minus_gen_op]

    def create_complex_basis_trf(self):
        generator_ops = self.create_complex_basis_gen_op()

        bases_trf = self.mx_op_builder.create_basis_trf(generator_ops, self.calc_order).truncate_matrix(self.trunc_num)

        return bases_trf
    
    def generate_new_bases(self):
        generator_ops = self.create_complex_basis_gen_op()
        return self.mx_op_builder.create_new_basis(generator_ops, self.calc_order-1)
    
    def generate_new_hilbert_space_and_bases(self):
        generator_ops = self.create_complex_basis_gen_op()

        return self.mx_op_builder.create_new_basis2(generator_ops, self.calc_order-1)
    

    def __init__(self,mode,spatial_dim, order, qm_nums_names, phonon_sys_name = '', id = ''):
        self.phonon_sys_name = phonon_sys_name
        self.mode = mode
        self.spatial_dim = spatial_dim
        self.order = order
        self.calc_order = order +1 
        self.qm_nums_names = qm_nums_names
        
        self.id = id
        self.children = []
        self.operators = {}

        self.calculation_bases = mm.hilber_space_bases().harm_osc_sys(self.spatial_dim,self.calc_order,qm_nums_names)

        self.hilbert_space_bases = self.calculation_bases
        

        self.calc_h_space_dim = self.calculation_bases.dim


        self.names_dict = { name:num for name,num in zip(self.qm_nums_names , range(0, len(self.qm_nums_names))) }
        self.mx_op_builder = mm.braket_to_matrix_formalism(self.calculation_bases)


        self.dim = self.calc_h_space_dim
        self.def_braket_create_qm_ops()
        self.def_braket_annil_qm_ops()


        self.calc_create_ops()
        self.calc_annil_ops()

        self.over_est_all_H_i_ops()
        self.over_est_H_op()
        self.trunc_num = self.calc_trunc_num()
        self.h_sp_dim = self.calc_h_space_dim-self.trunc_num
        self.mx_op_builder.used_dimension = self.h_sp_dim
        self.base_states = self.calculation_bases.reduce_space(self.h_sp_dim)

        self.dim  = self.h_sp_dim
        self.create_operators_dict()

    def create_operators_dict(self):
        self.operators['K'] = self.get_H_op()
    
        self.operators['X'] = self.calc_pos_i_op('x')
        self.operators['Y'] = self.calc_pos_i_op('y')

        self.operators['XX'] = self.calc_pos_i_j_op('x','x')
    
        self.operators['YY'] = self.calc_pos_i_j_op('y','y')

        self.operators['XY'] = self.calc_pos_i_j_op('x','y')

        self.operators['YX'] = self.calc_pos_i_j_op('y','x')


    def get_qm_num(self, state, key):
        if isinstance(state,bf.ket_state) or isinstance(state,bf.bra_state):
            qm_num_index = self.names_dict[key]
            return state.qm_state[ qm_num_index ]

    def def_braket_create_qm_ops(self):
        self.creator_braket_ops = []
        for key in self.names_dict.keys():
            creator_braket_op = bf.creator_operator(self.names_dict[key], key)
            self.creator_braket_ops.append(creator_braket_op)

    def def_braket_annil_qm_ops(self):
        self.annil_braket_ops = []
        for key in self.names_dict.keys():
            annil_braket_op = bf.annihilator_operator(self.names_dict[key],key)
            self.annil_braket_ops.append(annil_braket_op)

    def calc_create_ops(self):
        self.create_mx_ops = {}
        for creator_braket_op in self.creator_braket_ops:
            mx_op =  self.mx_op_builder.create_MatrixOperator(creator_braket_op, subsys_name = self.phonon_sys_name)
        
            self.create_mx_ops[creator_braket_op.name] = mx_op

    def calc_annil_ops(self):
        self.annil_mx_ops = {}
        for annil_braket_op in self.annil_braket_ops:
            mx_op =  self.mx_op_builder.create_MatrixOperator(annil_braket_op, subsys_name = self.phonon_sys_name)
            self.annil_mx_ops[annil_braket_op.name] = mx_op


    def over_est_H_i_op(self, qm_num_name):
        return self.create_mx_ops[qm_num_name]*self.annil_mx_ops[qm_num_name]

    def over_est_all_H_i_ops(self):
        self.H_i_ops = []
        for qm_nums_name in self.qm_nums_names:
            self.H_i_ops.append(self.create_mx_ops[qm_nums_name]*self.annil_mx_ops[qm_nums_name] + 0.5*self.create_id_op())


    def over_est_H_op(self):
        H = sum(self.H_i_ops)
        H.subsys_name = self.phonon_sys_name
        self.over_est_H = H.round(0).change_type(np.int16)

    def get_H_op(self ) -> mm.MatrixOperator:
        return self.mode*self.over_est_H.truncate_matrix(self.trunc_num)

    def calc_trunc_num(self):
        return self.over_est_H.matrix.count_occurrences(self.calc_order)

    def over_est_pos_i_j_op(self, qm_num_i, qm_num_j):
        pos_i_op = self.over_est_pos_i_op(qm_num_i)
        pos_j_op = self.over_est_pos_i_op(qm_num_j)

        return pos_i_op*pos_j_op


    def over_est_pos_i_op(self, qm_num_name) -> mm.MatrixOperator:
        op = ((self.annil_mx_ops[qm_num_name] + self.create_mx_ops[qm_num_name])
              / mm.SQRT_2)
        op.subsys_name = self.phonon_sys_name
        return op

    def calc_pos_prefactor(self):
        return 1.0
        # Alternative implementations (currently disabled):
        # return (self.K/2)**0.5
        # return (hbar_meVs/2)**0.5*self.K**-0.25
        # sqrt_2 = math.sqrt(2.0)
        # return (4.603275202*10**-13)/(sqrt_2*self.K**0.25)

    def calc_pos_i_op(self, qm_num_name) -> mm.MatrixOperator:

        return self.calc_pos_prefactor()*self.over_est_pos_i_op(qm_num_name).truncate_matrix(self.trunc_num)


    def calc_pos_i_j_op(self, qm_num_name_1, qm_num_name_2):

        return self.calc_pos_prefactor()**2*self.over_est_pos_i_j_op(qm_num_name_1, qm_num_name_2).truncate_matrix(self.trunc_num)



class Exe_tree:

    
    lambda_Ham:float = None
    delta_p_factor:float = None
    delta_f_factor:float = None
    KJT_factor:float = None
    lambda_SOC:float = None
    JT_theory:'jt.Jahn_Teller_Theory' = None
    lambda_theory:float = None
    electron:bool = False
    p_factor:float = None

    def get_lattice_basis_vecs(self):
        if self.JT_theory is not None and self.JT_theory.symm_lattice is not None:
            return self.JT_theory.symm_lattice.get_normalized_basis_vecs()
        else:
            return [maths.col_vector.from_list([1.0, 0.0, 0.0]), maths.col_vector.from_list([0.0, 1.0, 0.0]), maths.col_vector.from_list([0.0, 0.0, 1.0])]    
            #return self.JT_theory.JT_lattice.get_normalized_basis_vecs()
    def set_orientation_basis(self, basis_vectors:list[maths.col_vector]):


        self.basis_x = basis_vectors[0]
        self.basis_y = basis_vectors[1]
        self.basis_z = basis_vectors[2]

    def get_normalized_basis_vecs(self):
        return [self.basis_x.normalize(), self.basis_y.normalize(), self.basis_z.normalize()]

    @staticmethod
    def create_electron_phonon_Exe_tree(JT_theory,order, intrinsic_soc, orbital_red_fact, orientation_basis:list[maths.col_vector] = maths.cartesian_basis):
        """
        Create an Exe_tree from JT_theory. If JT_theory.order_flag == 0 (model Hamiltonian),
        creates a minimal_Exe_tree instead of a regular Exe_tree.
        
        Args:
            JT_theory: Jahn_Teller_Theory object
            order: Order parameter
            intrinsic_soc: Intrinsic spin-orbit coupling
            orbital_red_fact: Orbital reduction factor
            orientation_basis: List of column vectors defining orientation
            
        Returns:
            Exe_tree or minimal_Exe_tree depending on JT_theory.order_flag
        """
        # Check if this is a model Hamiltonian (order_flag == 0)
        if JT_theory.order_flag == 0:
            # Create minimal_Exe_tree for model Hamiltonian
            # Use orientation_basis from JT_theory if available, otherwise use provided one
            basis = getattr(JT_theory, 'orientation_basis', None) or orientation_basis
            JT_int = minimal_Exe_tree(basis, JT_theory)
            
            # Set parameters from JT_theory
            JT_int.lambda_theory = JT_theory.lambda_DFT
            JT_int.f_factor = JT_theory.f_factor
            JT_int.gL = JT_theory.gL
            JT_int.delta_f_factor = JT_theory.delta_f
            JT_int.Yx = getattr(JT_theory, 'Yx', 0.0)
            JT_int.Yy = getattr(JT_theory, 'Yy', 0.0)
            JT_int.orbital_red_fact = orbital_red_fact
            JT_int.intrinsic_soc = intrinsic_soc
            JT_int.electron = True if intrinsic_soc > 0.0 else False
            return JT_int
        
        # Regular Exe_tree creation for non-model Hamiltonians
        # Use system_builder module for system construction
        point_defect_tree = system_builder.build_electron_phonon_system(
            JT_theory, order
        )

        JT_int = Exe_tree(point_defect_tree, JT_theory, orientation_basis)

        JT_int.orbital_red_fact = orbital_red_fact
        JT_int.intrinsic_soc = intrinsic_soc
        JT_int.electron = True if intrinsic_soc > 0.0 else False
        return JT_int

    def add_spin_system(self):
        spin_sys = qs.quantum_system_node.create_spin_system_node()
        self.system_tree.insert_node('electron_system', spin_sys)

    @staticmethod
    def create_spin_electron_phonon_Exe_tree(JT_theory,order, intrinsic_soc, orbital_red_fact):
        """
        Create an Exe_tree with spin, electron, and phonon systems.
        
        This method now uses the models.system_builder module for system construction.
        
        Args:
            JT_theory: Jahn_Teller_Theory object
            order: Order parameter
            intrinsic_soc: Intrinsic spin-orbit coupling
            orbital_red_fact: Orbital reduction factor
            
        Returns:
            Exe_tree: Constructed execution tree
        """
        # Use system_builder module for system construction
        point_defect_tree = system_builder.build_spin_electron_phonon_system(
            JT_theory, order
        )

        JT_int = Exe_tree(point_defect_tree, JT_theory)

        JT_int.orbital_red_fact = orbital_red_fact
        JT_int.intrinsic_soc = intrinsic_soc
        JT_int.electron = True if intrinsic_soc > 0.0 else False
        return JT_int


    def calc_energy_splitting(self):
        self.lambda_theory = self.lambda_SOC+self.KJT_factor


    def calc_K_JT_factor(self):
        """
        Compute K_JT factor from Jahn-Teller interaction.
        
        This method now uses the models.reduction_factors module.
        """
        self.KJT_factor = compute_K_JT_factor(
            self.system_tree,
            self.H_int
        )

    def calc_reduction_factors(self):
        """
        Compute reduction factors for Jahn-Teller system.
        
        This method now uses the models.reduction_factors module.
        """
        result = compute_reduction_factors(
            self.system_tree,
            self.H_int,
            self.orbital_red_fact,
            self.intrinsic_soc
        )
        
        # Store results as instance attributes
        self.p_32 = result.p_32
        self.p_12 = result.p_12
        self.p_factor = result.p_factor
        self.lambda_Ham = result.lambda_Ham
        self.delta_p_factor = result.delta_p_factor
        self.f_factor = result.f_factor
        self.delta_f_factor = result.delta_f_factor
        self.lambda_SOC = result.lambda_SOC
        self.lambda_theory = result.lambda_theory


    # IO methods have been moved to io.file_io.results_formatter module
    # For backward compatibility, these methods delegate to the formatter
    def get_essential_theoretical_results_string(self):
        """
        Format theoretical results as a string.
        
        .. deprecated:: This method is kept for backward compatibility.
        Use io.file_io.results_formatter.format_theoretical_results_string() instead.
        """
        from jahn_teller_dynamics.io.file_io import results_formatter
        return results_formatter.format_theoretical_results_string(self)


    def get_essential_theoretical_results(self):
        """
        Format theoretical results as a dictionary.
        
        .. deprecated:: This method is kept for backward compatibility.
        Use io.file_io.results_formatter.format_theoretical_results_dict() instead.
        """
        from jahn_teller_dynamics.io.file_io import results_formatter
        return results_formatter.format_theoretical_results_dict(self)

    def save_essential_theoretical_results(self, res_path: str):
        """
        Save theoretical results to CSV file.
        
        .. deprecated:: This method is kept for backward compatibility.
        Use io.file_io.results_formatter.save_theoretical_results() instead.
        """
        from jahn_teller_dynamics.io.file_io import results_formatter
        results_formatter.save_theoretical_results(self, res_path)



    def get_essential_input_string(self):
        """
        Format input data as a string.
        
        .. deprecated:: This method is kept for backward compatibility.
        Use io.file_io.results_formatter.format_input_data_string() instead.
        """
        from jahn_teller_dynamics.io.file_io import results_formatter
        return results_formatter.format_input_data_string(self)


    def get_essential_input(self):
        """
        Format input data as a dictionary.
        
        .. deprecated:: This method is kept for backward compatibility.
        Use io.file_io.results_formatter.format_input_data_dict() instead.
        """
        from jahn_teller_dynamics.io.file_io import results_formatter
        return results_formatter.format_input_data_dict(self)
    def save_essential_input(self, res_folder: str, calc_name: str):
        """
        Save input data to CSV file.
        
        .. deprecated:: This method is kept for backward compatibility.
        Use io.file_io.results_formatter.save_input_data() instead.
        """
        from jahn_teller_dynamics.io.file_io import results_formatter
        res_path = res_folder + calc_name + '_essential_input.csv'
        results_formatter.save_input_data(self, res_path, calc_name=calc_name)

    def get_base_state(self):
        return self.system_tree.root_node.base_states

    def calc_eigen_vals_vects(self)->mm.eigen_vector_space:
        return self.H_int.calc_eigen_vals_vects(quantum_states_bases=self.system_tree.root_node.base_states)

    def save_eigen_vals_vects_to_file(self, eig_vec_fn, eig_val_fn):
        
        self.eig_vec_sys = self.H_int.calc_eigen_vals_vects(quantum_states_bases=self.system_tree.root_node.base_states)
        self.eig_vec_sys.save(eig_vec_fn, eig_val_fn)


    def __init__(self, system_tree: qs.quantum_system_tree, jt_theory:jt.Jahn_Teller_Theory, orientation_basis = maths.cartesian_basis):
        self.system_tree = system_tree
        self.JT_theory = jt_theory
        self.H_int: mm.MatrixOperator = None
        self.p_factor: float = None
        self.f_factor: float = None
        self.orbital_red_fact: float = None
        self.delta_p_factor: float = None
        self.KJT_factor: float = None
        self.intrinsic_soc: float = None
        self.lambda_Ham: float = None
        if orientation_basis is not None:
            self.set_orientation_basis(orientation_basis)
    
    def create_minimal_model_DJT_H_int(self, Bx, By, Bz):
        """
        Create minimal model DJT Hamiltonian with SOC and magnetic field.
        
        Formula:
            H = λ_full (L_z ⊗ S_z) + μ_B [f_factor * B_z * L_z + g_s * (S · B) + 2*δ_f * B_z * S_z]
        
        Where λ_full = -(λ_Ham + K_JT)
        
        This method now uses the hamiltonians modules.
        """
        lambda_full = -float((self.lambda_Ham + self.KJT_factor))
        
        # SOC term using spin_orbit module
        H_soc = spin_orbit.create_spin_orbit_coupling(
            self.system_tree,
            lambda_full
        )
        
        # Magnetic field interaction using field_interactions module
        H_mag = field_interactions.create_magnetic_field_interaction(
            self.system_tree,
            Bx, By, Bz,
            self.orbital_red_fact,
            self.f_factor if hasattr(self, 'f_factor') and self.f_factor is not None else self.orbital_red_fact,
            self.delta_f_factor if hasattr(self, 'delta_f_factor') and self.delta_f_factor is not None else 0.0
        )
        
        return H_soc + H_mag  

    def to_minimal_model(self,B_field):
        """
        Convert Exe_tree to minimal model (no phonons).
        
        This method now uses the models.system_builder module for system construction.
        
        Args:
            B_field: Magnetic field vector
            
        Returns:
            minimal_Exe_tree: Minimal model execution tree
        """
        new_obj = copy.deepcopy(self)

        # Use system_builder module for system construction
        new_obj.system_tree = system_builder.build_minimal_model_system()


        new_obj.intrinsic_soc = self.intrinsic_soc
        new_obj.H_int = new_obj.create_minimal_model_DJT_H_int(*B_field.tolist())

        return new_obj

    def create_spin_orbit_couping(self):
        """
        Create spin-orbit coupling operator: L_z ⊗ S_z.
        
        Note: This returns just the operator structure, not the full Hamiltonian.
        For the full SOC Hamiltonian with strength, use get_spin_orbit_coupling_int_ham().
        """
        Sz = self.system_tree.create_operator('Sz', 'spin_system')
        Lz = self.system_tree.create_operator('Lz', 'orbital_system')
        return Lz**Sz
    
    def add_spin_orbit_coupling(self):
        """
        Add spin-orbit coupling to the interaction Hamiltonian.
        
        This method now uses the models.operator_manager module.
        """
        LzSz_op = self.create_spin_orbit_couping()
        root_node_id = self.system_tree.root_node.id
        
        # Store operator and get root-level reference with coefficient
        soc_ham = add_operator_to_hamiltonian(
            self.system_tree,
            LzSz_op,
            'LzSz',
            'electron_system',
            'electron_system',
            coefficient=self.intrinsic_soc,
            root_node_id=root_node_id
        )
        
        self.H_int = self.H_int + soc_ham

    def get_spin_orbit_coupling_int_ham(self):
        """
        Get spin-orbit coupling interaction Hamiltonian.
        
        Formula:
            H_SOC = λ_intrinsic * (L_z ⊗ S_z)
        
        This method now uses the hamiltonians.spin_orbit module.
        """
        return spin_orbit.create_spin_orbit_coupling(
            self.system_tree,
            self.intrinsic_soc
        )


    def create_electric_field_interaction(self, E_x, E_y) -> mm.MatrixOperator:
        """
        Create electric field interaction Hamiltonian.
        
        Formula:
            H_el = E_x Z + E_y X
        
        This method now uses the hamiltonians.field_interactions module.
        """
        return field_interactions.create_electric_field_interaction(
            self.system_tree, E_x, E_y
        )
    
    def create_strain_field_interaction(self, strain_field: maths.col_vector) -> mm.MatrixOperator:
        """
        Create strain field interaction Hamiltonian.
        
        Formula:
            H_strain = -Y_x L_x + Y_y L_y + Y_z L_z
        
        This method now uses the hamiltonians.field_interactions module.
        """
        return field_interactions.create_strain_field_interaction(
            self.system_tree, strain_field
        )

    def create_magnetic_field_spin_z_interaction(self, B_z, delta, gl_factor) -> mm.MatrixOperator:
        """
        Create magnetic field spin-z interaction (anisotropic term).
        
        Formula:
            H_mag_spin_z = -2 * δ * g_L * μ_B * B_z * S_z
        
        This is a component of the full magnetic field interaction.
        Note: This method is kept for backward compatibility but the full
        magnetic field interaction should use create_magnetic_field_interaction().
        """
        from jahn_teller_dynamics.physics.hamiltonians.field_interactions import BOHR_MAGNETON_MEV_T
        
        Sz = self.system_tree.create_operator('Sz', 'spin_system')
        H_mag = -2 * delta * gl_factor * BOHR_MAGNETON_MEV_T * B_z * Sz
        return H_mag
    
    def create_magnetic_field_ang_interaction(self, B_z) -> mm.MatrixOperator:
        """
        Create magnetic field orbital interaction.
        
        Formula:
            H_mag_ang = μ_B * g_L * B_z * L_z
        
        Where g_L = orbital_red_fact
        Note: This method is kept for backward compatibility but the full
        magnetic field interaction should use create_magnetic_field_interaction().
        """
        from jahn_teller_dynamics.physics.hamiltonians.field_interactions import BOHR_MAGNETON_MEV_T
        
        Lz = self.system_tree.create_operator('Lz', 'orbital_system')
        H_mag = (BOHR_MAGNETON_MEV_T * self.orbital_red_fact) * B_z * Lz
        return H_mag

    def create_magnetic_field_spin_interaction(self, Bx, By, Bz) -> mm.MatrixOperator:
        """
        Create magnetic field spin interaction (isotropic term).
        
        Formula:
            H_mag_spin = μ_B * g_s * (B_x S_x + B_y S_y + B_z S_z)
        
        Where g_s = 2.0023 (electron spin g-factor)
        Note: This method is kept for backward compatibility but the full
        magnetic field interaction should use create_magnetic_field_interaction().
        """
        from jahn_teller_dynamics.physics.hamiltonians.field_interactions import BOHR_MAGNETON_MEV_T, G_FACTOR
        
        Sz = self.system_tree.create_operator('Sz', 'spin_system')
        Sy = self.system_tree.create_operator('Sy', 'spin_system')
        Sx = self.system_tree.create_operator('Sx', 'spin_system')

        return BOHR_MAGNETON_MEV_T * G_FACTOR * (Bx*Sx + By*Sy + Bz*Sz)

    def add_model_magnetic_field(self,Bz):

        root_node_id = self.system_tree.root_node.id
        Sz_point_def = self.system_tree.create_operator('H_mag_spin_z', subsys_id=root_node_id, operator_sys='spin_system')

        H_mag_model = self.pd*Bz*Sz_point_def

        self.H_int = self.H_int+H_mag_model


    def create_DJT_SOC_mag_interaction(self,Bx,By,Bz)->MatrixOperator:
        """
        Create DJT + SOC + magnetic field interaction Hamiltonian.
        
        Formula:
            H = H_DJT + H_SOC + H_mag_spin + H_mag_ang
        
        Where:
            - H_DJT: Dynamic Jahn-Teller Hamiltonian
            - H_SOC: Spin-orbit coupling (λ_intrinsic * L_z ⊗ S_z)
            - H_mag_spin: Magnetic field spin interaction (stored in spin_system)
            - H_mag_ang: Magnetic field orbital interaction (stored in orbital_system)
        
        Note: This method stores operators separately in subsystems for proper
        operator tree structure, which is different from create_magnetic_field_interaction().
        """
        self.create_one_mode_DJT_hamiltonian()
        H_DJT = self.system_tree.root_node.operators['H_DJT']

        H_full_int = H_DJT + self.get_spin_orbit_coupling_int_ham()

        # Get root node ID dynamically for reusability
        root_node_id = self.system_tree.root_node.id

        # Create and store spin interaction separately
        H_mag_spin = self.create_magnetic_field_spin_interaction(Bx, By, Bz)
        H_mag_spin_point_def = store_and_get_root_operator(
            self.system_tree,
            H_mag_spin,
            'H_mag_spin',
            'spin_system',
            'spin_system',
            root_node_id=root_node_id
        )

        # Create and store orbital interaction separately
        H_mag_ang = self.create_magnetic_field_ang_interaction(Bz)
        H_mag_ang_point_def = store_and_get_root_operator(
            self.system_tree,
            H_mag_ang,
            'H_mag_ang',
            'orbital_system',
            'orbital_system',
            root_node_id=root_node_id
        )

        return H_full_int + H_mag_spin_point_def + H_mag_ang_point_def 

    def calc_magnetic_interaction_eigen_kets(self, B_fields, strain_fields = None):
        """
        Compute eigenstates for different magnetic field configurations.
        
        This method now uses the numerics.observables module.
        """
        # Get basis vectors for field transformation
        normalized_basis_vecs = self.get_normalized_basis_vecs()
        symm_lattice_basis_vecs = None
        if self.JT_theory is not None and self.JT_theory.symm_lattice is not None:
            symm_lattice_basis_vecs = self.JT_theory.symm_lattice.get_normalized_basis_vecs()
        
        return compute_magnetic_interaction_eigen_kets(
            self.system_tree,
            B_fields,
            create_hamiltonian_func=self.create_DJT_SOC_mag_interaction,
            create_strain_func=self.create_strain_field_interaction if strain_fields is not None else None,
            strain_fields=strain_fields,
            normalized_basis_vecs=normalized_basis_vecs,
            symm_lattice_basis_vecs=symm_lattice_basis_vecs,
            num_states=4
        )

    def calc_transition_intensities(self, from_kets, to_kets):
        """
        Compute transition intensities between eigenstates.
        
        This method now uses the numerics.observables module.
        """
        return compute_transition_intensities(
            self.system_tree,
            from_kets,
            to_kets,
            include_z=False
        )

    def add_magnetic_field(self, Bx, By, Bz):
        """
        Add magnetic field interactions to the interaction Hamiltonian.
        
        This method now uses the models.operator_manager module.
        """
        root_node_id = self.system_tree.root_node.id

        # Create and store spin-z interaction
        H_mag_spin_z = self.create_magnetic_field_spin_z_interaction(Bz, self.delta_p_factor, self.orbital_red_fact)
        H_mag_spin_z_root = store_and_get_root_operator(
            self.system_tree,
            H_mag_spin_z,
            'H_mag_spin_z',
            'spin_system',
            'spin_system',
            root_node_id=root_node_id
        )

        # Create and store spin interaction
        H_mag_spin = self.create_magnetic_field_spin_interaction(Bx, By, Bz)
        H_mag_spin_root = store_and_get_root_operator(
            self.system_tree,
            H_mag_spin,
            'H_mag_spin',
            'spin_system',
            'spin_system',
            root_node_id=root_node_id
        )

        # Create and store orbital interaction
        H_mag_ang = self.create_magnetic_field_ang_interaction(Bz)
        H_mag_ang_root = store_and_get_root_operator(
            self.system_tree,
            H_mag_ang,
            'H_mag_ang',
            'orbital_system',
            'orbital_system',
            root_node_id=root_node_id
        )

        self.H_int = self.H_int + H_mag_spin_root + H_mag_ang_root + H_mag_spin_z_root

    def add_electric_field(self, E_x, E_y):
        """
        Add electric field interaction to the interaction Hamiltonian.
        
        This method now uses the models.operator_manager module.
        """
        H_el = self.create_electric_field_interaction(E_x, E_y)
        root_node_id = self.system_tree.root_node.id
        
        # Store operator and get root-level reference
        H_el_root = store_and_get_root_operator(
            self.system_tree,
            H_el,
            'H_el',
            'orbital_system',
            'orbital_system',
            root_node_id=root_node_id
        )
        
        self.H_int = self.H_int + H_el_root

    def create_multi_mode_hamiltonian(self):
        """
        Create multi-mode DJT Hamiltonian as sum over modes.
        
        Formula:
            H = Σ_i [K_i ⊗ I + F_i(X_i ⊗ σ_z + Y_i ⊗ σ_x) + G_i((XX_i-YY_i) ⊗ σ_z - 2XY_i ⊗ σ_x)]
        
        This method now uses the hamiltonians.djt_hamiltonian module.
        """
        # Use the new hamiltonian module
        self.H_int = djt_hamiltonian.create_multi_mode_djt_hamiltonian(
            self.system_tree,
            self.JT_theory
        )
        return self.H_int

    def create_one_mode_DJT_hamiltonian(self, mode=0.0):
        """
        Create one-mode DJT Hamiltonian.
        
        Formula:
            H = K ⊗ I + F(X ⊗ σ_z - Y ⊗ σ_x) + G((XX-YY) ⊗ σ_z + 2XY ⊗ σ_x)
        
        This method now uses the hamiltonians.djt_hamiltonian module
        for clearer physics implementation.
        """
        # Use the new hamiltonian module
        self.H_int = djt_hamiltonian.create_one_mode_djt_hamiltonian(
            self.system_tree, 
            self.JT_theory
        )
        
        # Calculate eigenvalues and store (preserve original behavior)
        self.H_int.calc_eigen_vals_vects(
            quantum_states_bases=self.system_tree.root_node.base_states
        )
        self.system_tree.root_node.operators['H_DJT'] = copy.deepcopy(self.H_int)




class minimal_Exe_tree(Exe_tree):

    

    @staticmethod
    def from_cfg_data(energy_split, orientation_basis, gL, delta_f, f_factor, Yx, Yy):
        
        
        tree = minimal_Exe_tree(orientation_basis)
        tree.Yx = Yx
        tree.Yy = Yy
        #tree.orbital_red_fact = orbital_red_fact
        tree.delta_f_factor = delta_f

        tree.lambda_theory = energy_split
        tree.f_factor = f_factor
        tree.gL = gL
        return tree


    def __init__(self,orientation_basis: list[maths.col_vector], jt_theory = None):
        """
        Initialize minimal Exe_tree for four-state model Hamiltonian.
        
        This method now uses the models.system_builder module for system construction.
        
        Args:
            orientation_basis: List of column vectors defining orientation
            jt_theory: Optional Jahn_Teller_Theory object
        """
        # Use system_builder module for system construction
        self.system_tree = system_builder.build_minimal_model_system()
        self.JT_theory = jt_theory
        self.H_int: mm.MatrixOperator = None
        self.p_factor: float = None
        self.f_factor: float = None
        self.orbital_red_fact: float = None
        self.delta_p_factor: float = None
        self.KJT_factor: float = None
        self.DFT_soc: float = None
        self.lambda_Ham: float = None
        self.Yx:float = 0.0
        self.Yy:float = 0.0
        self.electron = True
        self.set_orientation_basis(orientation_basis)

    def set_reduction_factors(self, exe_tree:Exe_tree):
        self.p_factor:float = exe_tree.p_factor
        self.f_factor:float = exe_tree.f_factor
        self.orbital_red_fact = exe_tree.orbital_red_fact
        self.delta_p_factor:float = exe_tree.delta_p_factor
        self.delta_f_factor:float = exe_tree.delta_f_factor
        self.KJT_factor:float = exe_tree.KJT_factor
        self.DFT_soc:float = exe_tree.intrinsic_soc
        self.electron = exe_tree.electron
        self.lambda_Ham:float = exe_tree.lambda_Ham
        self.lambda_theory = exe_tree.lambda_theory
        self.lambda_SOC = exe_tree.lambda_SOC
        self.intrinsic_soc = exe_tree.intrinsic_soc
        self.p_32 = exe_tree.p_32
        self.p_12 = exe_tree.p_12

    

    @staticmethod
    def from_Exe_tree(exe_tree:Exe_tree):
        model_exe_tree = minimal_Exe_tree([exe_tree.basis_x,exe_tree.basis_y,exe_tree.basis_z], exe_tree.JT_theory)
        model_exe_tree.set_reduction_factors(exe_tree)
        return model_exe_tree

    def create_spin_orbit_couping(self):
        """
        Create spin-orbit coupling operator: L_z ⊗ S_z at root node level.
        
        Override of parent method to create operators at root node level
        instead of subsystem level, for consistency with minimal_Exe_tree structure.
        """
        root_node_id = self.system_tree.root_node.id
        Sz = self.system_tree.create_operator('Sz', root_node_id, 'spin_system')
        Lz = self.system_tree.create_operator('Lz', root_node_id, 'orbital_system')
        return Lz*Sz

    def create_DJT_SOC_mag_interaction(self, Bx, By, Bz):
        """
        Create minimal model DJT + SOC + magnetic field interaction.
        
        Formula:
            H = λ_theory (L_z ⊗ S_z) + μ_B [f_factor * B_z * L_z + g_s * (S · B) + 2*δ_f * B_z * S_z]
        
        This method now uses the hamiltonians modules.
        """
        # SOC term using spin_orbit module
        H_soc = spin_orbit.create_spin_orbit_coupling(
            self.system_tree,
            self.lambda_theory
        )
        
        # Magnetic field interaction using field_interactions module
        H_mag = field_interactions.create_magnetic_field_interaction(
            self.system_tree,
            Bx, By, Bz,
            self.gL if hasattr(self, 'gL') and self.gL is not None else 0.0,
            self.f_factor if hasattr(self, 'f_factor') and self.f_factor is not None else 0.0,
            self.delta_f_factor if hasattr(self, 'delta_f_factor') and self.delta_f_factor is not None else 0.0
        )
        
        return H_soc + H_mag
        
    def create_one_mode_DJT_hamiltonian(self, mode=0):
        return MatrixOperator.create_null_matrix_op(dim = 4)

