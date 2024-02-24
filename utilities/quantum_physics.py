import utilities.matrix_formalism as mf
import utilities.braket_formalism as bf
from utilities.matrix_formalism import MatrixOperator
import itertools
from collections import namedtuple
import numpy as np
import utilities.jahn_teller_theory as  jt
import utilities.maths as maths
import utilities.quantum_system as qs
phonon_sys_data = namedtuple('phonon_sys_data', 'mode dim order qm_nums_names')

mode1data = phonon_sys_data(78, 2,5, [ 'mode_1_x, mode_1_y' ])



round_precision_dig = 7

class one_mode_phonon_sys(qs.quantum_system_node):

    def create_complex_basis_gen_op(self):

        raise_x_op = bf.raise_index_operator(0)

        raise_y_op = bf.raise_index_operator(1)

        raise_x_mx_op = self.mx_op_builder.create_MatrixOperator(raise_x_op).truncate_matrix(self.trunc_num)
        
        raise_y_mx_op = self.mx_op_builder.create_MatrixOperator(raise_y_op).truncate_matrix(self.trunc_num)




        plus_gen_op = 1/2**0.5*(raise_x_mx_op+complex(0.0,1.0)*raise_y_mx_op)
        minus_gen_op = 1/2**0.5*(raise_x_mx_op+complex(0.0,-1.0)*raise_y_mx_op)

        return [plus_gen_op, minus_gen_op]

    def create_complex_basis_trf(self):
        generator_ops = self.create_complex_basis_gen_op()

        #bases_trf = self.mx_op_builder.create_basis_trf(generator_ops, self.calc_order).truncate_matrix(self.trunc_num)
        bases_trf = self.mx_op_builder.create_basis_trf(generator_ops, self.calc_order).truncate_matrix(self.trunc_num)

        return bases_trf
    
    def generate_new_bases(self):
        generator_ops = self.create_complex_basis_gen_op()
        return self.mx_op_builder.create_new_basis(generator_ops, self.calc_order-1)
    
    def generate_new_hilbert_space_and_bases(self):
        generator_ops = self.create_complex_basis_gen_op()

        return self.mx_op_builder.create_new_basis2(generator_ops, self.calc_order-1)
    
        #return self.mx_op_builder.create_new_basis(generator_ops, self.calc_order-1)
        #return self.mx_op_builder.tree_basis_generation(generator_ops,self.calc_order)

    # Braket quantum mechanics
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

        self.calculation_bases = mf.hilber_space_bases().harm_osc_sys(self.spatial_dim,self.calc_order,qm_nums_names)

        self.hilbert_space_bases = self.calculation_bases
        

        self.calc_h_space_dim = self.calculation_bases.dim

        #Aggregation

        self.names_dict = { name:num for name,num in zip(self.qm_nums_names , range(0, len(self.qm_nums_names))) }
        self.mx_op_builder = mf.braket_to_matrix_formalism(self.calculation_bases)

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

        

    def get_signature(self) ->bf.quantum_subsystem_signature:
        return bf.quantum_subsystem_signature(self.phonon_sys_name, self.calc_h_space_dim-self.trunc_num, self.qm_nums_names)


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
            annil_braket_op = bf.annil_operator(self.names_dict[key],key)
            self.annil_braket_ops.append(annil_braket_op)

    # Matrix quantum mechanics
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
            self.H_i_ops.append(self.create_mx_ops[qm_nums_name]*self.annil_mx_ops[qm_nums_name])


    def over_est_H_op(self):
        H = sum(self.H_i_ops)
        H.subsys_name = self.phonon_sys_name
        self.over_est_H = H.round(0).change_type(np.int16)

    def get_H_op(self ) -> mf.MatrixOperator:
        return self.mode*self.over_est_H.truncate_matrix(self.trunc_num)

    def calc_trunc_num(self):
        return self.over_est_H.matrix.count_occurrences(self.calc_order)
        print('calc trunc num')

    def over_est_pos_i_j_op(self, qm_num_i, qm_num_j):
        pos_i_op = self.over_est_pos_i_op(qm_num_i)
        pos_j_op = self.over_est_pos_i_op(qm_num_j)

        return pos_i_op*pos_j_op


    def over_est_pos_i_op(self, qm_num_name) -> mf.MatrixOperator:
        op = ((self.annil_mx_ops[qm_num_name] + self.create_mx_ops[qm_num_name])
              /(2**0.5))
        op.subsys_name = self.phonon_sys_name
        return op

    def calc_pos_i_op(self, qm_num_name) -> mf.MatrixOperator:

        return self.over_est_pos_i_op(qm_num_name).truncate_matrix(self.trunc_num)
            


    def calc_pos_i_j_op(self, qm_num_name_1, qm_num_name_2):

        return self.over_est_pos_i_j_op(qm_num_name_1, qm_num_name_2).truncate_matrix(self.trunc_num)



class Exe_tree:
    def __init__(self, system_tree: qs.quantum_system_tree, jt_theory:jt.Jahn_Teller_Theory):
        self.system_tree = system_tree
        self.JT_theory = jt_theory
        self.H_int:mf.MatrixOperator
    
    def create_multi_mode_hamiltonian(self):

        hamiltons = []

        nuclei = self.system_tree.find_subsystem('nuclei')[0]

        for osc_mode in nuclei.children:
            osc_mode_id = osc_mode.id
            X = self.system_tree.create_operator('X', subsys_id= 'nuclei', operator_sys = osc_mode_id)
            Y = self.system_tree.create_operator('Y', subsys_id= 'nuclei', operator_sys = osc_mode_id)
            XX = self.system_tree.create_operator('XX', subsys_id= 'nuclei', operator_sys = osc_mode_id)
            YY = self.system_tree.create_operator('YY', subsys_id= 'nuclei', operator_sys = osc_mode_id)
            XY = self.system_tree.create_operator('XY', subsys_id= 'nuclei', operator_sys = osc_mode_id)
            YX = self.system_tree.create_operator('YX', subsys_id= 'nuclei', operator_sys = osc_mode_id)

            K = self.system_tree.create_operator('K', subsys_id= 'nuclei', operator_sys = osc_mode_id)

            self.JT_theory.set_quantum(osc_mode.mode)

            s0 = self.system_tree.create_operator('s0', 'electron_system')
            sz = self.system_tree.create_operator('sz', 'electron_system')
            sx = self.system_tree.create_operator('sx', 'electron_system')

            h =   K** s0 + self.JT_theory.F*(X**sz + Y**sx) + 1.0*self.JT_theory.G* ( (XX-YY) **sz - (2* XY)**sx)
            hamiltons.append(h)
        
        self.H_int = sum(hamiltons)
        return self.H_int

    def create_one_mode_hamiltonian(self, mode = 0.0):
        X = self.system_tree.create_operator('X', 'nuclei' )
        Y = self.system_tree.create_operator('Y', 'nuclei' )

        #print('X dim' +str( X.dim))
        XX = self.system_tree.create_operator('XX', 'nuclei' )
        YY = self.system_tree.create_operator('YY', 'nuclei' )
        XY = self.system_tree.create_operator('XY', 'nuclei' )
        YX = self.system_tree.create_operator('YX', 'nuclei' )

        K = self.system_tree.create_operator('K', 'nuclei')

        self.JT_theory.set_quantum(77.6)

        s0 = self.system_tree.create_operator('s0', 'electron_system')
        sz = self.system_tree.create_operator('sz', 'electron_system')
        sx = self.system_tree.create_operator('sx', 'electron_system')

        self.H_int =   K** s0 + self.JT_theory.F*(X**sz + Y**sx) + 1.0*self.JT_theory.G* ( (XX-YY) **sz - (2* XY)**sx)

        #return K** s0 + self.JT_theory.F*(X**sz + Y**sx) + 1.0*self.JT_theory.G* ( (XX-YY) **sz - (2* XY)**sx)

