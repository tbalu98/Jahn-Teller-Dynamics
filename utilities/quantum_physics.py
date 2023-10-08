import utilities.matrix_formalism as mf
import utilities.braket_formalism as bf
from utilities.matrix_formalism import MatrixOperator
import itertools
from collections import namedtuple
import numpy as np
import utilities.jahn_teller_theory as  jt
phonon_sys_data = namedtuple('phonon_sys_data', 'mode dim order qm_nums_names')

mode1data = phonon_sys_data(78, 2,5, [ 'mode_1_x, mode_1_y' ])





class one_mode_phonon_sys:

    # Braket quantum mechanics
    def __init__(self,mode,spatial_dim, order, qm_nums_names, phonon_sys_name = ''):
        self.phonon_sys_name = phonon_sys_name
        self.mode = mode
        self.spatial_dim = spatial_dim
        self.order = order
        self.calc_order = order +1 
        self.qm_nums_names = qm_nums_names

        self.phonon_states = bf.hilber_space_bases([],[]).harm_osc_sys(self.spatial_dim,self.calc_order)

        self.h_space_dim = self.phonon_states.dim

        #Aggregation

        self.names_dict = { name:num for name,num in zip(self.qm_nums_names , range(0, len(self.qm_nums_names))) }
        self.mx_op_builder = mf.braket_to_mx_operator_builder(self.phonon_states)

        self.def_braket_create_qm_ops()
        self.def_braket_annil_qm_ops()


        self.calc_create_ops()
        self.calc_annil_ops()

        self.over_est_all_H_i_ops()
        self.over_est_H_op()
        self.trunc_num = self.calc_trunc_num()

        

    def get_signature(self) ->bf.quantum_subsystem_signature:
        return bf.quantum_subsystem_signature(self.phonon_sys_name, self.h_space_dim-self.trunc_num, self.qm_nums_names)


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

    def get_H_op(self,qm_sys_sign:bf.quantum_system_signature == None ) -> mf.MatrixOperator:
        return self.over_est_H.truncate_matrix(self.trunc_num).as_part_of_a_system(qm_sys_sign)

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

    def calc_pos_i_op(self, qm_num_name, qm_sys_sign:bf.quantum_system_signature == None) -> mf.MatrixOperator:

        return self.over_est_pos_i_op(qm_num_name).truncate_matrix(self.trunc_num).as_part_of_a_system(qm_sys_sign)
            


    def calc_pos_i_j_op(self, qm_num_name_1, qm_num_name_2, qm_sys_sign:bf.quantum_system_signature==None):

        return self.over_est_pos_i_j_op(qm_num_name_1, qm_num_name_2).truncate_matrix(self.trunc_num).as_part_of_a_system(qm_sys_sign)

class multi_mode_phonon_system:
    
    def __init__(self, one_mode_phonon_systems: list[one_mode_phonon_sys]):
        self.one_mode_phonon_systems = one_mode_phonon_systems

        self.multi_mode_phonon_bases = self.create_hilbert_space()
        self.calc_signature()

    def calc_signature(self):
        self.signature =  bf.quantum_system_signature([ phonon_sys.get_signature() for phonon_sys in self.one_mode_phonon_systems ])

    def create_hilbert_space(self):
        h_spaces_to_kron = [ one_mode_phonon_system.phonon_states for one_mode_phonon_system in self.one_mode_phonon_systems ]
        self.phonon_states = bf.hilber_space_bases.kron_hilber_spaces(h_spaces_to_kron)

    def get_pos_i_j_op(self, sub_sys_name:str, qm_num_name_i:str, qm_num_name_j:str ):
        
        ph_sys = self.get_ph_sys(sub_sys_name)   
        return ph_sys.calc_pos_i_j_op(qm_num_name_i,qm_num_name_j).as_part_of_a_system(self.signature)

    def get_H_op(self, sub_sys_name:str):
        ph_sys = self.get_ph_sys(sub_sys_name)
        return ph_sys.get_H_op()


    def get_pos_i_op(self, sub_sys_name:str, qm_num_name:str )->mf.MatrixOperator:
        
        ph_sys = self.get_ph_sys(sub_sys_name)   
        return ph_sys.calc_pos_i_op(qm_num_name,self.signature)#.as_part_of_a_system(self.signature)

    def get_ph_sys(self, sub_sys_name:str):
        return list( filter( lambda x: x.phonon_sys_name==sub_sys_name ,self.one_mode_phonon_systems) )[0]


    def get_pos_i_op_old(self, sub_sys_name:str, qm_num_name:str):

        return self.calc_op( sub_sys_name, lambda x: x.calc_pos_i_op(qm_num_name) )




    def calc_op(self, subsys_name, subsys_op_getter ):
        ops_to_kron = []
        #ops_to_kron = [ subsys_op_getter(one_mode_phonon_system) if one_mode_phonon_system.phonon_sys_name == subsys_name 
        #               else mf.MatrixOperator.create_id_matrix_op(one_mode_phonon_system.h_space_dim) 
        #               for one_mode_phonon_system in self.one_mode_phonon_systems]

        for one_mode_phonon_system in self.one_mode_phonon_systems:
            if one_mode_phonon_system.phonon_sys_name == subsys_name:
                ops_to_kron.append(subsys_op_getter( one_mode_phonon_system ))
            else:
                ops_to_kron.append(mf.MatrixOperator.create_id_matrix_op(one_mode_phonon_system.h_space_dim))

        return mf.MatrixOperator.accumulate_operators(ops_to_kron, lambda x,y: x**y)

class electron_system:
    def __init__(self, h_space_bases:bf.hilber_space_bases, symmetries:dict[str:MatrixOperator] ):
        self.el_bases = h_space_bases
        self.symmetries = symmetries

class phonon_electron_system_one_modes:
    def __init__(self,ph_syss:list[one_mode_phonon_sys], el_sys:electron_system):
        self.ph_syss = ph_syss
        self.el_sys = el_sys
#        self.h_space_bases = bf.hilber_space_bases.kron_hilber_spaces([ph_syss.phonon_states,el_sys.el_bases])


class Exe_phonon_electron_system:
    def __init__(self, ph_sys:multi_mode_phonon_system,el_sys: electron_system, jt_theory:jt.Jahn_Teller_Theory):
        self.ph_sys = ph_sys
        self.el_sys = el_sys
        self.h_space_bases = bf.hilber_space_bases.kron_hilber_spaces([ph_sys.phonon_states,el_sys.el_bases])
        self.phonon_sys_sign = self.ph_sys.signature
        self.JT_theory = jt_theory
        self.create_Ham_op()

    def create_one_mode_H(self, one_mode_ph_sys:one_mode_phonon_sys):
        X = one_mode_ph_sys.calc_pos_i_op('x',self.phonon_sys_sign )
        Y = one_mode_ph_sys.calc_pos_i_op('y',self.phonon_sys_sign )

        XX = one_mode_ph_sys.calc_pos_i_j_op('x','x', self.phonon_sys_sign )
        YY = one_mode_ph_sys.calc_pos_i_j_op('y','y', self.phonon_sys_sign )

        XY = one_mode_ph_sys.calc_pos_i_j_op('x','y', self.phonon_sys_sign )
        YX = one_mode_ph_sys.calc_pos_i_j_op('y','x', self.phonon_sys_sign )

        K = one_mode_ph_sys.get_H_op(self.phonon_sys_sign)

        self.JT_theory.set_quantum(one_mode_ph_sys.mode)

        s0 = self.el_sys.symmetries['s0']
        sz = self.el_sys.symmetries['sz']
        sx = self.el_sys.symmetries['sx']

        return self.JT_theory.hw * K** s0 + self.JT_theory.F*(X**sz + Y**sx) + 1.0*self.JT_theory.G* ( (XX-YY) **sz - (XY + YX)**sx)


    def create_Ham_op(self):
        self.H_int = sum([ self.create_one_mode_H(one_mode_ph_sys) for one_mode_ph_sys in self.ph_sys.one_mode_phonon_systems ])


class quantum_sys_signature:

    def __init__(self, signatures:tuple[ str, int ]):
        self.signatures = signatures


