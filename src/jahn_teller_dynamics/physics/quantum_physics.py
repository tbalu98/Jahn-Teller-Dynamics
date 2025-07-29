import jahn_teller_dynamics.math.matrix_formalism as mf
import jahn_teller_dynamics.math.braket_formalism as bf
from jahn_teller_dynamics.math.matrix_formalism import MatrixOperator
import itertools
from collections import namedtuple
import numpy as np
import jahn_teller_dynamics.physics.jahn_teller_theory as  jt
import jahn_teller_dynamics.math.maths as maths
import jahn_teller_dynamics.physics.quantum_system as qs
import copy
import jahn_teller_dynamics.io.VASP as VASP
import pandas as pd



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

        self.calculation_bases = mf.hilber_space_bases().harm_osc_sys(self.spatial_dim,self.calc_order,qm_nums_names)

        self.hilbert_space_bases = self.calculation_bases
        

        self.calc_h_space_dim = self.calculation_bases.dim


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
            #self.H_i_ops.append(self.create_mx_ops[qm_nums_name]*self.annil_mx_ops[qm_nums_name] )


    def over_est_H_op(self):
        H = sum(self.H_i_ops)
        H.subsys_name = self.phonon_sys_name
        self.over_est_H = H.round(0).change_type(np.int16)

    def get_H_op(self ) -> mf.MatrixOperator:
        return self.mode*self.over_est_H.truncate_matrix(self.trunc_num)

    def calc_trunc_num(self):
        return self.over_est_H.matrix.count_occurrences(self.calc_order)

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

    
    lambda_Ham:float = None
    delta_p_factor:float = None
    delta_f_factor:float = None
    KJT_factor:float = None
    lambda_SOC:float = None
    JT_theory:jt.Jahn_Teller_Theory = None
    lambda_theory:float = None

    def set_orientation_basis(self, basis_vectors:list[maths.col_vector]):
        """
        if self.JT_theory.symm_lattice.basis_vecs!=None:
            
            self.basis_x = basis_vectors[0].in_new_basis(self.JT_theory.symm_lattice.basis_vecs)
            self.basis_y = basis_vectors[1].in_new_basis(self.JT_theory.symm_lattice.basis_vecs)
            self.basis_z = basis_vectors[2].in_new_basis(self.JT_theory.symm_lattice.basis_vecs)
        """

        self.basis_x = basis_vectors[0]
        self.basis_y = basis_vectors[1]
        self.basis_z = basis_vectors[2]

    def get_normalized_basis_vecs(self):
        return [self.basis_x.normalize(), self.basis_y.normalize(), self.basis_z.normalize()]

    def create_electron_phonon_Exe_tree(JT_theory,order, intrinsic_soc, orbital_red_fact, orientation_basis:list[maths.col_vector] = maths.cartesian_basis):
        
        spatial_dim = 2


        orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()

        electron_system = qs.quantum_system_node('electron_system', children=[orbital_system])


        mode_1 = one_mode_phonon_sys(JT_theory.hw_meV,spatial_dim,order,['x','y'], 'mode_1', 'mode_1' )


        nuclei = qs.quantum_system_node('nuclei')

        point_defect_node = qs.quantum_system_node('point_defect', 
                                                children = [ nuclei,electron_system])

        point_defect_tree = qs.quantum_system_tree(point_defect_node)

        point_defect_tree.insert_node('nuclei', mode_1)


        JT_int =  Exe_tree(point_defect_tree, JT_theory, orientation_basis)

        JT_int.orbital_red_fact = orbital_red_fact
        JT_int.intrinsic_soc = intrinsic_soc
        return JT_int

    def add_spin_system(self):
        spin_sys = qs.quantum_system_node.create_spin_system_node()
        self.system_tree.insert_node('electron_system', spin_sys)

    def create_spin_electron_phonon_Exe_tree(JT_theory,order, intrinsic_soc, orbital_red_fact):
        spatial_dim = 2


        orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()

        electron_system = qs.quantum_system_node('electron_system', children=[orbital_system])

        spin_sys = qs.quantum_system_node.create_spin_system_node()

        mode_1 = one_mode_phonon_sys(JT_theory.hw_meV,spatial_dim,order,['x','y'], 'mode_1', 'mode_1' )


        nuclei = qs.quantum_system_node('nuclei')

        point_defect_node = qs.quantum_system_node('point_defect', 
                                                children = [ nuclei,electron_system])

        point_defect_tree = qs.quantum_system_tree(point_defect_node)

        point_defect_tree.insert_node('nuclei', mode_1)

        point_defect_tree.insert_node('electron_system', spin_sys)

        JT_int =  Exe_tree(point_defect_tree, JT_theory)

        JT_int.orbital_red_fact = orbital_red_fact
        JT_int.intrinsic_soc = intrinsic_soc
        return JT_int


    def calc_energy_splitting(self):
        self.lambda_theory = self.lambda_SOC+self.KJT_factor


    def calc_K_JT_factor(self):

        if self.H_int.eigen_kets!=None:
            H_DJT =  self.system_tree.root_node.operators['H_DJT']
            
            E_32 = self.H_int.eigen_kets[3]
            E_12 = self.H_int.eigen_kets[0]

            K_JT_32 =  H_DJT.calc_expected_val(E_32)
            K_JT_12 =  H_DJT.calc_expected_val(E_12)
            self.KJT_factor = K_JT_32-K_JT_12

    def calc_reduction_factors(self):
        LzSz_op = self.system_tree.create_operator('LzSz',subsys_id='point_defect', operator_sys='electron_system')
        self.p_32 = 2*LzSz_op.calc_expected_val(self.H_int.eigen_kets[2])
        self.p_12 = -2*LzSz_op.calc_expected_val(self.H_int.eigen_kets[0])

        self.p_factor = abs((self.p_32+self.p_12)/2)
        self.lambda_Ham = self.p_factor

        self.delta_p_factor = (self.p_32-self.p_12)/2

        self.f_factor = self.orbital_red_fact*self.p_factor

        self.delta_f_factor = self.orbital_red_fact*self.delta_p_factor

        self.lambda_SOC = self.p_factor*self.intrinsic_soc

        self.lambda_theory = self.H_int.eigen_kets[2].eigen_val- self.H_int.eigen_kets[0].eigen_val.real


    def get_essential_theoretical_results_string(self):
        res_str = 'Theoretical results:\n'

        res_str+='\n\tHam reduction factor = ' + str(round(self.p_factor,4)) if self.p_factor != None else ''
        res_str+='\n\tTheoretical energy level splitting = ' + str(round(self.lambda_theory,4)) + ' meV'

  
        return res_str


    def get_essential_theoretical_results(self):
        
        temp_res_dict = self.get_essential_input()



        if self.JT_theory.order_flag == 1 or self.JT_theory.order_flag == 2:
            temp_res_dict['Jahn-Teller energy (meV)'] = [self.JT_theory.E_JT_meV]
        if self.JT_theory.order_flag==2:
            temp_res_dict['barrier energy (meV)'] = [self.JT_theory.delta_meV]


        
        temp_res_dict['vibrational energy quantum (meV)'] = [self.JT_theory.hw_meV]
        temp_res_dict['Ham reduction factor'] = [self.p_factor]
        
        temp_res_dict['delta_p factor'] = [self.delta_p_factor] if self.delta_f_factor !=None else[ None]
        temp_res_dict['delta_f factor'] = [self.delta_f_factor] if self.delta_f_factor !=None else [None]
        temp_res_dict['f factor'] = [self.f_factor] if self.delta_f_factor!=None else [None]
        temp_res_dict['Energy splitting due to dynamic Janh-Teller effect (meV)'] = [self.KJT_factor] if self.KJT_factor!=None else [None]

        temp_res_dict['Energy splitting due to spin-orbit coupling (meV)'] = [self.lambda_SOC] if self.lambda_SOC!=None else [None]
        temp_res_dict['Energy splitting (meV)'] = [self.lambda_theory] if self.lambda_theory!=None else [None]

        res_dict = {}

        res_dict[ 'attribute'] = [ str(x) for x in  temp_res_dict.keys()]

        res_dict['values'] = [ str(x[0]) for x in temp_res_dict.values() ]

        return res_dict

    def save_essential_theoretical_results(self, res_path:str):
        res_dict = self.get_essential_theoretical_results()
        res_df = pd.DataFrame(res_dict).set_index('attribute')
        res_df.to_csv(res_path)



    def get_essential_input_string(self):
        res_str = 'Input data from ab initio calculations:\n'

        res_str += '\tsymmetric geometry energy = ' + str(round(self.JT_theory.symm_lattice.energy,4))+ ' eV' +'\n'
        res_str += '\tminimum geometry energy '+ ' = ' +str( round(self.JT_theory.JT_lattice.energy,4))+ ' eV' +'\n'
        res_str += '\tsaddle point geometry energy = ' +str( round(self.JT_theory.barrier_lattice.energy,4))+ ' eV' +'\n' if self.JT_theory.order_flag==2 else ''

        res_str+= '\tsymmetric - minimum geometry distance = ' + str( round(self.JT_theory.JT_dist,4)) + ' Å √amu ' +'\n'
        res_str+= '\tsymmetric - saddle point geometry distance = '+str( round(self.JT_theory.barrier_dist,4)) + ' Å √amu ' +'\n' if self.JT_theory.order_flag==2 else ''
        res_str+= '\DFT spin-orbit coupling = ' + str(round(self.intrinsic_soc,4))+ ' meV' +'\n'
        res_str+='\torbital reduction factor = '+ str(round(self.orbital_red_fact,4)) +'\n'

        return res_str


    def get_essential_input(self):
        res_dict = {}

        if self.JT_theory.order_flag!= 3:
            res_dict['symmetric geometry energy (eV)'] = [self.JT_theory.symm_lattice.energy] if self.JT_theory.symm_lattice!=None else [None]
            res_dict['minimum geometry energy (eV)'] = [self.JT_theory.JT_lattice.energy] if self.JT_theory.JT_lattice!=None else [None]
            res_dict['saddle point geometry energy (eV)'] = [self.JT_theory.barrier_lattice.energy] if self.JT_theory.barrier_lattice!= None and self.JT_theory.order_flag==2 else [None]

        if self.JT_theory.order_flag == 1 or self.JT_theory.order_flag == 2:
            res_dict['high symmetry - minimum energy configuration distance (Å √amu)'] = [self.JT_theory.JT_dist]
        if self.JT_theory.order_flag==2:
            res_dict['high symmetry - saddle point configuration distance (Å √amu)'] = [self.JT_theory.barrier_dist]


        res_dict['DFT spin-orbit coupling (meV)'] = [self.intrinsic_soc]
        res_dict['orbital reduction factor '] = [self.orbital_red_fact]

        return res_dict

    def save_essential_input(self,  res_folder:str,calc_name:str):
        input_data_res = self.get_essential_input()


        input_data_res['calculation name'] = [calc_name]


        input_data_df = pd.DataFrame(input_data_res).set_index('calculation name')

        input_data_df.to_csv(res_folder + calc_name +'_essential_input.csv')


    def get_base_state(self):
        return self.system_tree.root_node.base_states

    def calc_eigen_vals_vects(self)->mf.eigen_vector_space:
        return self.H_int.calc_eigen_vals_vects(quantum_states_bases=self.system_tree.root_node.base_states)

    def save_eigen_vals_vects_to_file(self, eig_vec_fn, eig_val_fn):
        
        self.eig_vec_sys = self.H_int.calc_eigen_vals_vects(quantum_states_bases=self.system_tree.root_node.base_states)
        self.eig_vec_sys.save(eig_vec_fn, eig_val_fn)


    def __init__(self, system_tree: qs.quantum_system_tree, jt_theory:jt.Jahn_Teller_Theory, orientation_basis = maths.cartesian_basis):
        self.system_tree = system_tree
        self.JT_theory = jt_theory
        self.H_int:mf.MatrixOperator
        self.p_factor:float
        self.f_factor:float
        self.orbital_red_fact:float
        self.delta_p_factor:float
        self.KJT_factor:float
        self.intrinsic_soc:float
        self.lambda_Ham:float
        self.set_orientation_basis(orientation_basis)
    
    def create_minimal_model_DJT_H_int(self, Bx, By, Bz):

        Lz = self.system_tree.create_operator('Lz','point_defect','orbital_system')

        Sz = self.system_tree.create_operator('Sz', 'point_defect','spin_system')
        Sy = self.system_tree.create_operator('Sy', 'point_defect','spin_system')
        Sx = self.system_tree.create_operator('Sx', 'point_defect','spin_system')


        lambda_full = -float((self.lambda_Ham + self.KJT_factor))

        return lambda_full*self.create_spin_orbit_couping() + Bohn_magneton_meV_T*self.f_factor*Bz*Lz + Bohn_magneton_meV_T*g_factor*( Bx*Sx + By*Sy+ Bz*Sz  ) + 2*Bohn_magneton_meV_T*self.delta_f_factor*Bz*Sz
        

    def to_minimal_model(self,B_field):
    

        new_obj = copy.deepcopy(self)

        orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()
        spin_sys = qs.quantum_system_node.create_spin_system_node()

        electron_system = qs.quantum_system_node('electron_system', children=[orbital_system, spin_sys])

        point_defect = qs.quantum_system_node('point_defect', children = [electron_system])

        new_obj.system_tree = qs.quantum_system_tree(point_defect)


        new_obj.intrinsic_soc = self.intrinsic_soc
        new_obj.H_int = new_obj.create_minimal_model_DJT_H_int(*B_field.tolist())

        return new_obj

    def create_spin_orbit_couping(self):

        Sz = self.system_tree.create_operator('Sz', 'spin_system')
        Lz = self.system_tree.create_operator('Lz', 'orbital_system')
        return Lz**Sz
    
    def add_spin_orbit_coupling(self):
        
        LzSz_op = self.create_spin_orbit_couping()
        self.system_tree.find_subsystem('electron_system').operators['LzSz'] = LzSz_op

        self.H_int = self.H_int+self.intrinsic_soc*self.system_tree.create_operator('LzSz',subsys_id='point_defect', operator_sys='electron_system')

    def get_spin_orbit_coupling_int_ham(self):
        return self.intrinsic_soc*self.system_tree.create_operator('LzSz',subsys_id='point_defect', operator_sys='electron_system')


    def create_electric_field_interaction(self, E_x, E_y)->mf.MatrixOperator:

        Z = self.system_tree.create_operator('Z_orb', 'orbital_system')
        X = self.system_tree.create_operator('X_orb', 'orbital_system')

        H_el = E_x*Z + E_y*X


        return H_el
    
    def create_strain_field_interaction(self, Stx, Sty, Stz):
        pass

    def create_magnetic_field_spin_z_interaction(self, B_z, delta, gl_factor)->mf.MatrixOperator:

        Sz = self.system_tree.create_operator('Sz', 'spin_system')


        H_mag = -2*delta*gl_factor*Bohn_magneton_meV_T*B_z*Sz
        
        return H_mag
    
    def create_magnetic_field_ang_interaction(self, B_z)->mf.MatrixOperator:

        Lz = self.system_tree.create_operator('Lz', 'orbital_system')

        H_mag = (Bohn_magneton_meV_T*self.orbital_red_fact)*B_z*Lz
        
        return H_mag

    def create_magnetic_field_spin_interaction(self, Bx, By, Bz)->mf.MatrixOperator:

        Sz = self.system_tree.create_operator('Sz', 'spin_system')
        Sy = self.system_tree.create_operator('Sy', 'spin_system')
        Sx = self.system_tree.create_operator('Sx', 'spin_system')

        return Bohn_magneton_meV_T * g_factor*(Bx*Sx + By*Sy + Bz*Sz)

    def add_model_magnetic_field(self,Bz):

        Sz_point_def = self.system_tree.create_operator('H_mag_spin_z', subsys_id = 'point_defect', operator_sys='spin_system')

        H_mag_model = self.pd*Bz*Sz_point_def

        self.H_int = self.H_int+H_mag_model


    def create_DJT_SOC_mag_interaction(self,Bx,By,Bz)->MatrixOperator:
        
        
        H_DJT = self.system_tree.root_node.operators['H_DJT']



        H_full_int = H_DJT + self.get_spin_orbit_coupling_int_ham()

        H_mag_spin = self.create_magnetic_field_spin_interaction(Bx, By, Bz)
        self.system_tree.find_subsystem('spin_system').operators['H_mag_spin'] = H_mag_spin
        H_mag_spin_point_def = self.system_tree.create_operator('H_mag_spin', subsys_id = 'point_defect', operator_sys='spin_system')

        H_mag_ang = self.create_magnetic_field_ang_interaction( Bz)

        self.system_tree.find_subsystem('orbital_system').operators['H_mag_ang'] = H_mag_ang
        H_mag_ang_point_def = self.system_tree.create_operator('H_mag_ang', subsys_id = 'point_defect', operator_sys='orbital_system')

        return H_full_int + H_mag_spin_point_def + H_mag_ang_point_def 

    def add_magnetic_field(self, Bx,By,Bz):



        H_mag_spin_z = self.create_magnetic_field_spin_z_interaction(Bz, self.delta_p_factor, self.orbital_red_fact)
        self.system_tree.find_subsystem('spin_system').operators['H_mag_spin_z'] = H_mag_spin_z
        H_mag_spin_z_point_def = self.system_tree.create_operator('H_mag_spin_z', subsys_id='point_defect', operator_sys='spin_system')

        H_mag_spin = self.create_magnetic_field_spin_interaction(Bx, By, Bz)
        self.system_tree.find_subsystem('spin_system').operators['H_mag_spin'] = H_mag_spin
        H_mag_spin_point_def = self.system_tree.create_operator('H_mag_spin', subsys_id = 'point_defect', operator_sys='spin_system')

        H_mag_ang = self.create_magnetic_field_ang_interaction( Bz, self.f_factor)

        self.system_tree.find_subsystem('orbital_system').operators['H_mag_ang'] = H_mag_ang
        H_mag_ang_point_def = self.system_tree.create_operator('H_mag_ang', subsys_id = 'point_defect', operator_sys='orbital_system')

        self.H_int = self.H_int + H_mag_spin_point_def + H_mag_ang_point_def + H_mag_spin_z_point_def

    """
    def create_spin_orbit_in_mag_field_ham(self, Bx, By, Bz):
        
        Lz = self.system_tree.create_operator('Lz', 'orbital_system')
        Sz = self.system_tree.create_operator('Sz', 'spin_system')

        H_SO = self.intrinsic_soc*Lz**Sz


        H_mag_spin_z = self.create_magnetic_field_spin_z_interaction(Bz, self.delta_p_factor, self.orbital_red_fact)
        self.system_tree.find_subsystem('spin_system').operators['H_mag_spin_z'] = H_mag_spin_z
        H_mag_spin_z_el_sys = self.system_tree.create_operator('H_mag_spin_z', subsys_id='electron_system', operator_sys='spin_system')

        H_mag_spin = self.create_magnetic_field_spin_interaction(Bx, By, Bz)
        self.system_tree.find_subsystem('spin_system').operators['H_mag_spin'] = H_mag_spin
        H_mag_spin_el_sys = self.system_tree.create_operator('H_mag_spin', subsys_id = 'electron_system', operator_sys='spin_system')

        H_mag_ang = self.create_magnetic_field_ang_interaction( Bz, self.f_factor)
        self.system_tree.find_subsystem('orbital_system').operators['H_mag_ang'] = H_mag_ang
        H_mag_ang_el_sys = self.system_tree.create_operator('H_mag_ang', subsys_id = 'electron_system', operator_sys='orbital_system')


        return  H_SO + H_mag_spin_z_el_sys + H_mag_spin_el_sys + H_mag_ang_el_sys
    """

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

        XX = self.system_tree.create_operator('XX', 'nuclei' )
        YY = self.system_tree.create_operator('YY', 'nuclei' )
        XY = self.system_tree.create_operator('XY', 'nuclei' )
        YX = self.system_tree.create_operator('YX', 'nuclei' )

        K = self.system_tree.create_operator('K', 'nuclei')

        s0 = self.system_tree.find_subsystem('electron_system').create_id_op()
        sz = self.system_tree.create_operator('Z_orb', 'electron_system')
        sx = self.system_tree.create_operator('X_orb', 'electron_system')



        self.H_int =   K** s0 + self.JT_theory.F*(X**sz + Y**sx) + 1.0*self.JT_theory.G* ( (XX-YY) **sz - (2* XY)**sx)
        #self.H_int =    self.JT_theory.F*(X**sz + Y**sx) + 1.0*self.JT_theory.G* ( (XX-YY) **sz - (2* XY)**sx)
        
        self.H_int.calc_eigen_vals_vects()
        self.system_tree.root_node.operators['H_DJT'] = copy.deepcopy(self.H_int)




class minimal_Exe_tree(Exe_tree):

    def from_cfg_data( orbital_red_fact, delta_p_factor, orientation_basis:list[maths.col_vector],p_factor = None, dft_soc = None,KJT = None,f_factor = None, soc_split_en = None ):
        tree = minimal_Exe_tree(orientation_basis)
        tree.p_factor = p_factor
        tree.DFT_soc = dft_soc
        tree.orbital_red_fact = orbital_red_fact
        tree.KJT_factor = KJT
        tree.delta_p_factor = delta_p_factor
        tree.lambda_theory = p_factor*dft_soc if p_factor is not None and dft_soc is not None else None

        if KJT != None and p_factor!= None and dft_soc != None:
            tree.lambda_theory = tree.p_factor*tree.DFT_soc + tree.KJT_factor
            tree.f_factor = tree.p_factor*tree.orbital_red_fact
            tree.lambda_SOC = tree.p_factor*tree.DFT_soc
        elif soc_split_en!= 0.0:
            tree.lambda_theory = soc_split_en
            tree.f_factor = f_factor
        
        tree.delta_f_factor = tree.delta_p_factor*tree.orbital_red_fact
        #tree.set_orientation_basis(orientation_basis)
        return tree


    def __init__(self,orientation_basis: list[maths.col_vector], jt_theory = None):
        orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()
        spin_sys = qs.quantum_system_node.create_spin_system_node()

        electron_system = qs.quantum_system_node('electron_system', children=[orbital_system, spin_sys])

        point_defect_system = qs.quantum_system_node( 'point_defect', children= [electron_system])

        system_tree = qs.quantum_system_tree(point_defect_system)

        self.system_tree = system_tree
        self.JT_theory = jt_theory
        self.H_int:mf.MatrixOperator
        self.p_factor:float
        self.f_factor:float
        self.orbital_red_fact:float
        self.delta_p_factor:float
        self.KJT_factor:float
        self.DFT_soc:float
        self.lambda_Ham:float
        self.set_orientation_basis(orientation_basis)

    def set_reduction_factors(self, exe_tree:Exe_tree):
        self.p_factor:float = exe_tree.p_factor
        self.f_factor:float = exe_tree.f_factor
        self.orbital_red_fact = exe_tree.orbital_red_fact
        self.delta_p_factor:float = exe_tree.delta_p_factor
        self.delta_f_factor:float = exe_tree.delta_f_factor
        self.KJT_factor:float = exe_tree.KJT_factor
        self.DFT_soc:float = exe_tree.intrinsic_soc
        self.lambda_Ham:float = exe_tree.lambda_Ham
        self.lambda_theory = exe_tree.lambda_theory
        self.lambda_SOC = exe_tree.lambda_SOC

    

    def from_Exe_tree(exe_tree:Exe_tree):
        model_exe_tree = minimal_Exe_tree([exe_tree.basis_x,exe_tree.basis_y,exe_tree.basis_z], exe_tree.JT_theory)
        model_exe_tree.set_reduction_factors(exe_tree)
        return model_exe_tree


    def create_DJT_SOC_mag_interaction(self, Bx, By, Bz):
        Lz = self.system_tree.create_operator('Lz','point_defect','orbital_system')

        Sz = self.system_tree.create_operator('Sz', 'point_defect','spin_system')
        Sy = self.system_tree.create_operator('Sy', 'point_defect','spin_system')
        Sx = self.system_tree.create_operator('Sx', 'point_defect','spin_system')


        #lambda_full = float((self.lambda_SOC + self.KJT_factor)) 

        return self.lambda_theory*self.create_spin_orbit_couping() + Bohn_magneton_meV_T*self.f_factor*Bz*Lz + Bohn_magneton_meV_T*g_factor*( Bx*Sx + By*Sy+ Bz*Sz  ) + 2*Bohn_magneton_meV_T*self.delta_f_factor*Bz*Sz
        
    def create_one_mode_DJT_hamiltonian(self, mode=0):
        return 

