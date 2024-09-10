import utilities.matrix_formalism as mf
import utilities.braket_formalism as bf
from utilities.matrix_formalism import MatrixOperator
import itertools
from collections import namedtuple
import numpy as np
import utilities.jahn_teller_theory as  jt
import utilities.maths as maths
import utilities.quantum_system as qs
import copy
phonon_sys_data = namedtuple('phonon_sys_data', 'mode dim order qm_nums_names')

mode1data = phonon_sys_data(78, 2,5, [ 'mode_1_x, mode_1_y' ])

Bohn_magneton_meV_T = 1.0
Bohn_magneton_meV_T = 0.057883671

g_factor = 2.0023

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

        
    """
    def get_signature(self) ->bf.quantum_subsystem_signature:
        return bf.quantum_subsystem_signature(self.phonon_sys_name, self.calc_h_space_dim-self.trunc_num, self.qm_nums_names)
    """

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
            self.H_i_ops.append(self.create_mx_ops[qm_nums_name]*self.annil_mx_ops[qm_nums_name] + 0.5*self.create_id_op())


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



    def calc_K_JT_factor(self):

        if self.H_int.eigen_kets!=None:
            
            E_32 = self.H_int.eigen_kets[2]
            E_12 = self.H_int.eigen_kets[0]

            K_JT_32 =  self.H_int.calc_expected_val(E_32)
            K_JT_12 =  self.H_int.calc_expected_val(E_12)
            self.KJT_factor = K_JT_12-K_JT_32

    def calc_reduction_factors(self):
        LzSz_op = self.system_tree.create_operator('LzSz',subsys_id='point_defect', operator_sys='electron_system')
        self.p_32 = 2*LzSz_op.calc_expected_val(self.H_int.eigen_kets[2])
        self.p_12 = -2*LzSz_op.calc_expected_val(self.H_int.eigen_kets[0])

        self.p_factor = (self.p_32+self.p_12)/2
        self.delta_p_factor = (self.p_32-self.p_12)/2

        self.f_factor = self.gL_factor*self.p_factor

        self.delta_f_factor = self.gL_factor*self.delta_p_factor

        self.lambda_Ham = self.H_int.eigen_kets[2].eigen_val- self.H_int.eigen_kets[0].eigen_val.real


    def get_essential_theoretical_results_string(self):
        res_str = 'Theoretical results:\n'

        res_str+='\n\tHam reduction factor = ' + str(round(self.p_factor,4)) 
        res_str+='\n\tTheoretical spin-orbit coupling = ' + str(round(self.lambda_Ham,4)) + ' meV'

        """
        res_str+='\n\tp_12 = ' + str(round(self.p_12,4))
        res_str+='\n\tp_32 = ' + str(round(self.p_32,4))
        
        res_str+='\n\tdelta_p factor =' + str(round(self.delta_p_factor,4)) 
        res_str+='\n\tdelta_f_factor =' + str(round(self.delta_f_factor, 4)) 
        res_str+='\n\tf factor = ' + str(round(self.f_factor,4))
        res_str+='\n\tK_JT factor = ' + str(round(self.KJT_factor,4)) + ' meV'
        """
        return res_str


    def get_essential_theoretical_results(self):
        
        res_dict = {}

        res_dict['Jahn-Teller energy (meV)'] = [self.JT_theory.E_JT_meV]
        if self.JT_theory.order_flag==2:
            res_dict['barrier energy (meV)'] = [self.JT_theory.delta_meV]
        res_dict['vibrational energy quantum (meV)'] = [self.JT_theory.hw_meV]
        res_dict['Ham reduction factor'] = [self.p_factor]
        res_dict['Theoretical spin-orbit coupling (meV)'] = [self.lambda_Ham]
        
        res_dict['delta_p factor'] = [self.delta_p_factor]
        res_dict['delta_f factor'] = [self.delta_f_factor]
        res_dict['f factor'] = [self.f_factor]
        res_dict['K_JT factor (meV)'] = [self.KJT_factor]

        return res_dict

    def get_essential_input_string(self):
        res_str = 'Input data from ab initio calculations:\n'

        res_str += '\tsymmetric lattice energy = ' + str(round(self.JT_theory.symm_lattice.energy,4))+ ' eV' +'\n'
        res_str += '\tJahn-Teller distorted lattice energy '+ ' = ' +str( round(self.JT_theory.JT_lattice.energy,4))+ ' eV' +'\n'
        res_str += '\tBarrier distorted lattice energy = ' +str( round(self.JT_theory.barrier_lattice.energy,4))+ ' eV' +'\n' if self.JT_theory.order_flag==2 else ''

        res_str+= '\tsymmetric lattice - Jahn-Teller distorted lattice distance = ' + str( round(self.JT_theory.JT_dist,4)) + ' Å √amu ' +'\n'
        res_str+= '\tsymmetric lattice - barrier distorted lattice distance = '+str( round(self.JT_theory.barrier_dist,4)) + ' Å √amu ' +'\n' if self.JT_theory.order_flag==2 else ''
        res_str+= '\tintrinsic spin-orbit coupling = ' + str(round(self.lambda_factor,4))+ ' meV' +'\n'
        res_str+='\torbital reduction factor = '+ str(round(self.gL_factor,4)) +'\n'

        return res_str


    def get_essential_input(self):
        res_dict = {}

        res_dict['symmetric lattice energy (eV)'] = self.JT_theory.symm_lattice.energy
        res_dict['Jahn-Teller distorted lattice energy (eV)'] = self.JT_theory.JT_lattice.energy
        res_dict['Barrier distorted lattice energy (eV)'] = self.JT_theory.barrier_lattice.energy if self.JT_theory.order_flag==2 else None

        res_dict['symmetric lattice - Jahn-Teller distorted lattice distance (Å √amu)'] = self.JT_theory.JT_dist
        res_dict['symmetric lattice - barrier distorted lattice distance (Å √amu)'] = self.JT_theory.barrier_dist if self.JT_theory.order_flag==2 else None
        res_dict['intrinsic spin-orbit coupling (meV)'] = self.lambda_factor
        res_dict['orbital reduction factor '] = [self.gL_factor]

        return res_dict



    def get_base_state(self):
        return self.system_tree.root_node.base_states

    def save_eigen_vals_vects_to_file(self, eig_vec_file_name, eig_val_file_name):
        self.H_int.save_eigen_vals_vects_to_file(self.system_tree.root_node.base_states,eig_vec_file_name, eig_val_file_name)

    def __init__(self, system_tree: qs.quantum_system_tree, jt_theory:jt.Jahn_Teller_Theory):
        self.system_tree = system_tree
        self.JT_theory = jt_theory
        self.H_int:mf.MatrixOperator
        self.p_factor:float
        self.f_factor:float
        self.gL_factor:float
        self.delta_p_factor:float
        self.KJT_factor:float
        self.lambda_factor:float
        self.lambda_Ham:float
    
    def create_minimal_model_DJT_H_int(self):
        #H_ZJT=ZJTx*kron(eye(l),sz)+ZJTy*kron(eye(l),sx); #zeroJT for Hepp dissertation

        """
        sz = self.system_tree.create_operator('Z_orb', 'electron_system')
        sx = self.system_tree.create_operator('X_orb', 'electron_system')
        return self.ZJTx*sz + self.ZJTy*sx
        """
    
        return  mf.MatrixOperator.create_null_matrix_op (dim = self.system_tree.root_node.dim)

    def to_minimal_model(self,ZJTx, ZJTy):
        self.ZJTx = ZJTx
        self.ZJTy = ZJTy

        new_obj = copy.deepcopy(self)

        orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()
        spin_sys = qs.quantum_system_node.create_spin_system_node()

        electron_system = qs.quantum_system_node('electron_system', children=[orbital_system, spin_sys])

        point_defect = qs.quantum_system_node('point_defect', children = [electron_system])

        new_obj.system_tree = qs.quantum_system_tree(point_defect)

        #new_obj.lambda_factor = self.lambda_factor*self.p_factor#+self.KJT_factor

        new_obj.lambda_factor = self.lambda_factor
        new_obj.H_int = self.create_minimal_model_DJT_H_int()

        return new_obj

    def create_spin_orbit_couping(self):

        Sz = self.system_tree.create_operator('Sz', 'spin_system')
        Lz = self.system_tree.create_operator('Lz', 'orbital_system')
        #Sz = self.system_tree.find_subsystem('spin_system').create_id_op()
        return Lz**Sz
    
    def add_spin_orbit_coupling(self):
        
        LzSz_op = self.create_spin_orbit_couping()
        self.system_tree.find_subsystem('electron_system').operators['LzSz'] = LzSz_op

        self.H_int = self.H_int+self.lambda_factor*self.system_tree.create_operator('LzSz',subsys_id='point_defect', operator_sys='electron_system')




    def create_electric_field_interaction(self, E_x, E_y)->mf.MatrixOperator:

        #Lz = self.system_tree.create_operator('Lz', 'orbital_system')
        #Lx = self.system_tree.create_operator('sx', 'orbital_system')

        Z = self.system_tree.create_operator('Z_orb', 'orbital_system')
        X = self.system_tree.create_operator('X_orb', 'orbital_system')

        H_el = E_x*Z + E_y*X


        return H_el
    
    

    def create_magnetic_field_spin_z_interaction(self, B_z, delta, gl_factor)->mf.MatrixOperator:

        Sz = self.system_tree.create_operator('Sz', 'spin_system')

        #H_mag = -2*delta*B_z*Sz
        #H_mag = -2*delta*gl_factor*B_z*Sz
        H_mag = -2*delta*gl_factor*Bohn_magneton_meV_T*B_z*Sz
        
        return H_mag
    
    def create_magnetic_field_ang_interaction(self, B_z, f)->mf.MatrixOperator:

        Lz = self.system_tree.create_operator('Lz', 'orbital_system')

        H_mag = -2*(Bohn_magneton_meV_T*f)*B_z*Lz
        
        return H_mag

    def create_magnetic_field_spin_interaction(self, Bx, By, Bz)->mf.MatrixOperator:

        Sz = self.system_tree.create_operator('Sz', 'spin_system')
        Sy = self.system_tree.create_operator('Sy', 'spin_system')
        Sx = self.system_tree.create_operator('Sx', 'spin_system')

        return Bohn_magneton_meV_T * g_factor*(Bx*Sx + By*Sy + Bz*Sz)

    def add_model_magnetic_field(self,Bz):
        #2*pd*(Bz*kron(kron(eye(l) ,s0),0.5*Sz))

        Sz_point_def = self.system_tree.create_operator('H_mag_spin_z', subsys_id = 'point_defect', operator_sys='spin_system')

        H_mag_model = self.pd*Bz*Sz_point_def

        self.H_int = self.H_int+H_mag_model

    def add_magnetic_field(self, Bx,By,Bz):

        H_mag_spin_z = self.create_magnetic_field_spin_z_interaction(Bz, self.delta_p_factor, self.gL_factor)
        self.system_tree.find_subsystem('spin_system').operators['H_mag_spin_z'] = H_mag_spin_z
        H_mag_spin_z_point_def = self.system_tree.create_operator('H_mag_spin_z', subsys_id='point_defect', operator_sys='spin_system')

        H_mag_spin = self.create_magnetic_field_spin_interaction(Bx, By, Bz)
        self.system_tree.find_subsystem('spin_system').operators['H_mag_spin'] = H_mag_spin
        H_mag_spin_point_def = self.system_tree.create_operator('H_mag_spin', subsys_id = 'point_defect', operator_sys='spin_system')

        H_mag_ang = self.create_magnetic_field_ang_interaction( Bz, self.f_factor)
        self.system_tree.find_subsystem('orbital_system').operators['H_mag_ang'] = H_mag_ang
        H_mag_ang_point_def = self.system_tree.create_operator('H_mag_ang', subsys_id = 'point_defect', operator_sys='orbital_system')

        self.H_int = self.H_int + H_mag_spin_point_def + H_mag_ang_point_def + H_mag_spin_z_point_def

    def create_spin_orbit_in_mag_field_ham(self, Bx, By, Bz):
        
        #Spin-orbit Hamiltonian
        Lz = self.system_tree.create_operator('Lz', 'orbital_system')
        Sz = self.system_tree.create_operator('Sz', 'spin_system')

        H_SO = self.lambda_factor*Lz**Sz

        #Add magnetic field

        H_mag_spin_z = self.create_magnetic_field_spin_z_interaction(Bz, self.delta_p_factor, self.gL_factor)
        self.system_tree.find_subsystem('spin_system').operators['H_mag_spin_z'] = H_mag_spin_z
        H_mag_spin_z_el_sys = self.system_tree.create_operator('H_mag_spin_z', subsys_id='electron_system', operator_sys='spin_system')

        H_mag_spin = self.create_magnetic_field_spin_interaction(Bx, By, Bz)
        self.system_tree.find_subsystem('spin_system').operators['H_mag_spin'] = H_mag_spin
        H_mag_spin_el_sys = self.system_tree.create_operator('H_mag_spin', subsys_id = 'electron_system', operator_sys='spin_system')

        H_mag_ang = self.create_magnetic_field_ang_interaction( Bz, self.f_factor)
        self.system_tree.find_subsystem('orbital_system').operators['H_mag_ang'] = H_mag_ang
        H_mag_ang_el_sys = self.system_tree.create_operator('H_mag_ang', subsys_id = 'electron_system', operator_sys='orbital_system')


        return  H_SO + H_mag_spin_z_el_sys + H_mag_spin_el_sys + H_mag_ang_el_sys


    def add_electric_field(self, E_x, E_y):
        H_el = self.create_electric_field_interaction(E_x, E_y)
        self.system_tree.find_subsystem('orbital_system').operators['H_el'] = H_el

        self.H_int = self.H_int + self.system_tree.create_operator('H_el', subsys_id='point_defect', operator_sys='orbital_system')

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

    def create_one_mode_DJT_hamiltonian(self, mode = 0.0):
        X = self.system_tree.create_operator('X', 'nuclei' )
        Y = self.system_tree.create_operator('Y', 'nuclei' )

        #print('X dim' +str( X.dim))
        XX = self.system_tree.create_operator('XX', 'nuclei' )
        YY = self.system_tree.create_operator('YY', 'nuclei' )
        XY = self.system_tree.create_operator('XY', 'nuclei' )
        YX = self.system_tree.create_operator('YX', 'nuclei' )

        K = self.system_tree.create_operator('K', 'nuclei')

        #self.JT_theory.set_quantum(77.6)

        #s0 = self.system_tree.create_operator('s0', 'electron_system')
        s0 = self.system_tree.find_subsystem('electron_system').create_id_op()
        sz = self.system_tree.create_operator('Z_orb', 'electron_system')
        sx = self.system_tree.create_operator('X_orb', 'electron_system')


        
        self.H_int =   K** s0 + self.JT_theory.F*(X**sz + Y**sx) + 1.0*self.JT_theory.G* ( (XX-YY) **sz - (2* XY)**sx)
        
        #self.H_int = 1000*self.system_tree.root_node.create_id_op()
        #self.H_int =    self.JT_theory.F*(X**sz + Y**sx) + 1.0*self.JT_theory.G* ( (XX-YY) **sz - (2* XY)**sx)

        #return K** s0 + self.JT_theory.F*(X**sz + Y**sx) + 1.0*self.JT_theory.G* ( (XX-YY) **sz - (2* XY)**sx)


"""
class minimal_Exe_model(Exe_tree):
    def __init__(self, jt_theory):
        orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()
        spin_sys = qs.quantum_system_node.create_spin_system_node()

        electron_system = qs.quantum_system_node('electron_system', children=[orbital_system, spin_sys])

        super.__init__(qs.quantum_system_tree(electron_system),jt_theory)
        self.H_int = maths.Matrix.create_zeros(dim = self.system_tree.root_node.dim)

"""
