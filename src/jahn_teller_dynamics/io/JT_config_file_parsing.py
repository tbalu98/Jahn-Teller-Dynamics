from configparser import ConfigParser
import jahn_teller_dynamics.physics.jahn_teller_theory as jt
import jahn_teller_dynamics.io.VASP as V
import jahn_teller_dynamics.math.maths as maths
import numpy as np
import jahn_teller_dynamics.physics.quantum_physics as qmp
from collections import namedtuple
import jahn_teller_dynamics.io.xml_parser as xml_parser
import os


#Fields:
essentials_field = 'essentials'
so_c_field = 'spin_orbit_coupling'
mag_field = 'magnetic_field'
csv_field = '.csv_files'
xml_field = 'vasprun.xml_files'
el_field = 'electric_field'
ex_state_field = 'excited_state_parameters'
gnd_state_field = 'ground_state_parameters'
system_field = 'system_parameters'
strain_field = 'strain_field'
#Options:
at_pat_opt = 'atom_parameters'
out_folder_opt = 'output_folder'
in_folder_opt  = 'input_folder'
spectrum_range_opt = 'spectrum_range'
orb_red_fact_op = 'orbital_reduction_factor'
delta_f_opt = 'delta_f'
out_prefix_opt = 'output_prefix'
int_soc_opt = 'DFT_spin-orbit_coupling'
symm_latt_opt = 'high_symmetry_geometry'
JT_latt_opt = 'global_minimum_geometry'
barr_latt_opt = 'saddle_point_geometry'
dir_opt = 'direction vector'
from_opt = 'from'
to_opt = 'to'
step_num_opt = 'step_number'
max_vib_quant = 'maximum_number_of_vibrational_quanta'
EJT_opt = 'Jahn-Teller_energy'
F_opt='F'
G_opt='G'
E_barr_opt = 'barrier_energy'
basis_vector_1_opt = 'basis_vector_1'
basis_vector_2_opt = 'basis_vector_2'
basis_vector_3_opt = 'basis_vector_3'
num_of_atoms_op = 'numbers_of_atoms'
mass_of_atoms_op = 'masses_of_atoms'
names_of_atoms_op = 'names_of_atoms'
hw_opt = 'vibrational_energy_quantum'
min_field_opt = 'minimum'
max_field_opt = 'maximum'
symm_min_dist_opt = 'high_symmetric_geometry-minimum_energy_geometry_distance'
symm_saddl_dist_opt = 'high_symmetric_geometry-saddle_point_geometry_distance'
save_raw_pars_opt = 'save_raw_parameters'
symm_latt_energy_opt = 'high_symmetric_geometry_energy'
min_energy_latt_energy_opt = 'global_minimum_energy'
saddle_point_latt_energy_opt = 'saddle_point_energy'
dir_vec_opt = 'direction_vector'
atom_structure_field = 'atom_structure_parameters'
mag_field_strength_csv_col =  'magnetic field (T)'
single_case_section = 'system_parameters'
eigen_states_opt = 'eigen_states'
model_Hamiltonian_opt = 'model_Hamiltonian'
orientation_vector_base = 'orientation_vector'
Ham_red_opt = 'Ham_reduction_factor'
f_factor_opt = 'f'
delta_p_opt = 'delta_p'
Yx_opt = 'Yx'
Yy_opt = 'Yy'
strain_vec_op = 'strain_vector'
K_JT_opt = 'K_JT'
save_model_Hamiltonian_cfg_opt = 'save_model_Hamiltonian_cfg'
save_Taylor_coeffs_cfg_opt =  'save_taylor_coeffs_cfg'
SOC_split_opt = 'spin-orbit_splitting_energy'
con_int_en_opt = 'conical_intersection_energy'
con_int_loc_opt = 'conical_intersection_location'
global_min_loc_opt = 'global_minimum_location'
saddle_point_loc_opt = 'saddle_point_location'


class Jahn_Teller_config_parser:

    def read_empty_lattice(self):
        pass

    def is_real_eigen_vects(self):
        return True if self.get_option_of_field(essentials_field,eigen_states_opt ) == 'real' else False

    def is_complex_eigen_vects(self):
        return True if self.get_option_of_field(essentials_field,eigen_states_opt ) == 'complex' else False


    def build_jt_theory_from_vasprunxmls(self,data_folder, section_to_look_for):

        symm_latt_fn = data_folder + self.get_option_of_field(section_to_look_for, symm_latt_opt)
        JT_geom_fn =data_folder + self.get_option_of_field(section_to_look_for, JT_latt_opt)
        barr_geom_fn =data_folder + self.get_option_of_field(section_to_look_for, barr_latt_opt) if self.config.has_option(section_to_look_for ,barr_latt_opt) else None
        
        
        symm_geom = xml_parser.xml_parser( symm_latt_fn).lattice
        JT_geom = xml_parser.xml_parser( JT_geom_fn ).lattice
        barr_geom = xml_parser.xml_parser( barr_geom_fn ).lattice if barr_geom_fn!=None else None
        JT_theory = jt.Jahn_Teller_Theory(symm_geom, JT_geom, barr_geom)
        
        return JT_theory

    def create_Jahn_Teller_theory_from_cfg(self,section_to_look_for):
        data_folder = self.get_data_folder_name()
        if self.is_from_JT_pars(section_to_look_for):
            JT_theory = self.build_JT_theories_from_cfg(section_to_look_for)

        elif self.is_from_vasprun_xml(section_to_look_for):
            JT_theory = self.build_jt_theory_from_vasprunxmls(data_folder, section_to_look_for)

        elif self.is_from_csv(section_to_look_for):
            JT_theory = self.build_jt_theory_from_csv_and_pars_data( section_to_look_for )

        elif self.is_from_Taylor_coeffs(section_to_look_for):
            JT_theory = self.build_JT_theory_from_Taylor_coeffs(section_to_look_for)

        elif self.is_from_model_Hamiltonian(section_to_look_for):
            JT_theory = self.build_JT_theory_from_model_Hamiltonian_cfg(section_to_look_for)

        elif self.is_from_energy_distance_pairs(section_to_look_for):
            JT_theory = self.build_JT_theories_from_energy_distance_pairs(section_to_look_for)

        return JT_theory
    


    def get_magnetic_field_vectors(self):


        if self.config.has_section(mag_field):
            #Bs = self.get_mag_field_strengths_list()

            Bs = self.get_mag_field_strengths_list()

            basis_vectors = self.get_basis_col_vectors(mag_field)

            dir_vec = self.get_mag_dir_vector().normalize()

            B_fields =  [ B*dir_vec for B in Bs ]
            if basis_vectors != None:

                B_fields = [ B_field.in_new_basis(basis_vectors) for B_field in B_fields ]
            return B_fields
        else:
            return None
        
    def get_field_vectors(self, field_name):

        if self.config.has_section(field_name):

            field_strengths = self.get_field_strengths_list()

            basis_vectors = self.get_basis_col_vectors(field_name)

            dir_vec = self.get_field_dir().normalize()

            field_vecs =  [ F*dir_vec for F in field_strengths ]
            if basis_vectors != None:

                field_vecs = [ B_field.in_new_basis(basis_vectors) for B_field in field_vecs ]
            return field_vecs
        else:
            return None

    def get_strain_field_vector(self) -> maths.col_vector:
        if self.config.has_section(strain_field):
            return self.get_strain_dir_vector()
        else:
            return None

    def get_calc_LzSz(self):
        res_str = self.get_option_of_field(essentials_field, spectrum_range_opt)
        return int(res_str) if res_str!='' else 0

    def is_single_case(self):
        return self.config.has_section(single_case_section)

    def save_raw_pars_section(self, JT_int:qmp.Exe_tree, section_to_cfg):

        state_parameters_section = {}

        state_parameters_section[int_soc_opt] = self.get_spin_orbit_coupling(section_to_cfg)
        state_parameters_section[orb_red_fact_op] = self.get_gL_factor(section_to_cfg)
        

        problem_name = self.get_prefix_name()
        symm_geom = JT_int.JT_theory.symm_lattice
        saddle_point_geom = JT_int.JT_theory.barrier_lattice if JT_int.JT_theory.barrier_lattice != None else None
        min_energy_geom = JT_int.JT_theory.JT_lattice
        data_folder = self.get_data_folder_name()

        df =  symm_geom.to_coordinates_data_frame()

        filenamebase = problem_name + '_' + section_to_cfg 

        symm_latt_geom_filename = filenamebase+'_'+ symm_latt_opt +'.csv'
        df.to_csv(data_folder +   symm_latt_geom_filename, sep = ';')
        state_parameters_section[symm_latt_opt] = symm_latt_geom_filename
        
        state_parameters_section[symm_latt_energy_opt] = str(symm_geom.energy)

        less_symm_latt_geom_filename_1 = filenamebase+'_'+JT_latt_opt+'.csv'
        df =  min_energy_geom.to_coordinates_data_frame()
        df.to_csv(data_folder + less_symm_latt_geom_filename_1, sep = ';')
        state_parameters_section[JT_latt_opt] = less_symm_latt_geom_filename_1
        state_parameters_section[min_energy_latt_energy_opt] = str(min_energy_geom.energy)


        if saddle_point_geom!=None:

            state_parameters_section[saddle_point_latt_energy_opt] = str(saddle_point_geom.energy)
            df =  saddle_point_geom.to_coordinates_data_frame()
            barrier_lattice_filename = filenamebase+'_' + barr_latt_opt+'.csv'
            df.to_csv( data_folder + barrier_lattice_filename, sep = ';')
            state_parameters_section[barr_latt_opt] = barrier_lattice_filename


        return state_parameters_section

        

    def save_raw_pars_ZPL(self, JT_int_gnd:qmp.Exe_tree, JT_int_ex:qmp.Exe_tree):

        data_folder = self.get_data_folder_name()
        problem_name = self.get_prefix_name()

        new_ZPL_config = ConfigParser()


        new_ZPL_config[gnd_state_field] = self.save_raw_pars_section(JT_int_gnd, gnd_state_field)
        new_ZPL_config[ex_state_field] = self.save_raw_pars_section(JT_int_ex, ex_state_field)
        new_ZPL_config[atom_structure_field] = JT_int_ex.JT_theory.symm_lattice.create_atom_pars_dict()


        new_ZPL_config[essentials_field] = self.config[essentials_field]
        new_ZPL_config[essentials_field][save_raw_pars_opt] = 'false'
        new_ZPL_config[essentials_field][model_Hamiltonian_opt] = 'false'
        new_ZPL_config[essentials_field][save_model_Hamiltonian_cfg_opt] = 'false'

        new_ZPL_config = self.add_mag_field_to_cfg(new_ZPL_config, JT_int_gnd)

        

        csv_cfg_name = os.path.join(self.config_file_dir, problem_name+'_csv.cfg')
        with open( csv_cfg_name, 'w') as xml_conf:
            new_ZPL_config.write(xml_conf)



    def save_raw_pars_ZPL_model(self, JT_int_gnd: qmp.Exe_tree, JT_int_ex: qmp.Exe_tree):
        
        JT_int_gnd = qmp.minimal_Exe_tree.from_Exe_tree(JT_int_gnd)
        JT_int_ex = qmp.minimal_Exe_tree.from_Exe_tree(JT_int_ex)


        
        problem_name = self.get_prefix_name()

        new_ZPL_config = ConfigParser()
        #essentials
        new_ZPL_config[essentials_field] = self.config[essentials_field]
        new_ZPL_config[essentials_field][save_raw_pars_opt] = 'false'
        new_ZPL_config[essentials_field][model_Hamiltonian_opt] = 'false'
        new_ZPL_config[essentials_field][save_model_Hamiltonian_cfg_opt] = 'false'

        new_ZPL_config[gnd_state_field] = self.save_model_raw_pars_section(JT_int_gnd, gnd_state_field)
        new_ZPL_config[ex_state_field] = self.save_model_raw_pars_section(JT_int_ex, ex_state_field)




        new_ZPL_config = self.add_mag_field_to_cfg(new_ZPL_config, JT_int_gnd)

        model_cfg_name = os.path.join(self.config_file_dir, problem_name+'_model.cfg')
        with open( model_cfg_name, 'w') as xml_conf:
            new_ZPL_config.write(xml_conf)

        #new_ZPL_config[mag_field][basis_vector_1_opt] = self.config[mag_field][ba]

    def add_mag_field_to_cfg(self, new_config:ConfigParser, JT_int:qmp.Exe_tree):

        new_config.add_section(mag_field)
        new_config[mag_field][min_field_opt] = self.config[mag_field][min_field_opt]
        new_config[mag_field][max_field_opt] = self.config[mag_field][max_field_opt]
        new_config[mag_field][step_num_opt] = self.config[mag_field][step_num_opt]
        new_config[mag_field][dir_vec_opt] = self.config[mag_field][dir_vec_opt]

        basis_vectors = self.get_basis_col_vectors(mag_field)
        if basis_vectors != None:
            new_config[mag_field][basis_vector_1_opt] = str(basis_vectors[0])
            new_config[mag_field][basis_vector_2_opt] = str(basis_vectors[1])
            new_config[mag_field][basis_vector_3_opt] = str(basis_vectors[2])
        elif JT_int.JT_theory.symm_lattice.basis_vecs != None:
            pass

        return new_config

    def save_raw_pars_ZPL_Taylor(self, JT_int_gnd: qmp.Exe_tree, JT_int_ex: qmp.Exe_tree):
        
        if type(JT_int_ex) is qmp.minimal_Exe_tree or type(JT_int_gnd) is qmp.minimal_Exe_tree:
            return
        
        problem_name = self.get_prefix_name()

        new_ZPL_config = ConfigParser()
        #essentials
        new_ZPL_config[essentials_field] = self.config[essentials_field]
        new_ZPL_config[essentials_field][save_raw_pars_opt] = 'false'
        new_ZPL_config[essentials_field][model_Hamiltonian_opt] = 'false'
        new_ZPL_config[essentials_field][save_model_Hamiltonian_cfg_opt] = 'false'
        new_ZPL_config[essentials_field][save_Taylor_coeffs_cfg_opt] = 'false'

        new_ZPL_config[gnd_state_field] = self.save_Taylor_raw_pars_section(JT_int_gnd, gnd_state_field)
        new_ZPL_config[ex_state_field] = self.save_Taylor_raw_pars_section(JT_int_ex, ex_state_field)


        #save magnetic field
        new_ZPL_config.add_section(mag_field)
        new_ZPL_config[mag_field][min_field_opt] = self.config[mag_field][min_field_opt]
        new_ZPL_config[mag_field][max_field_opt] = self.config[mag_field][max_field_opt]
        new_ZPL_config[mag_field][step_num_opt] = self.config[mag_field][step_num_opt]
        new_ZPL_config[mag_field][dir_vec_opt] = self.config[mag_field][dir_vec_opt]

        new_ZPL_config[mag_field][basis_vector_1_opt] = str(JT_int_gnd.JT_theory.symm_lattice.basis_vecs[0])
        new_ZPL_config[mag_field][basis_vector_2_opt] = str(JT_int_gnd.JT_theory.symm_lattice.basis_vecs[1])
        new_ZPL_config[mag_field][basis_vector_3_opt] = str(JT_int_gnd.JT_theory.symm_lattice.basis_vecs[2])



        taylor_cfg_name = os.path.join(self.config_file_dir, problem_name+'_Taylor_coeffs.cfg')
        with open( taylor_cfg_name, 'w') as xml_conf:
            new_ZPL_config.write(xml_conf)




    def save_model_raw_pars_section(self, JT_int: qmp.minimal_Exe_tree, section_to_cfg):
        state_parameters_section = {}
        #state_parameters_section[SOC_split_opt] = self.get_spin_orbit_coupling(section_to_cfg)
        state_parameters_section[SOC_split_opt] = JT_int.lambda_theory if JT_int.electron == True else -JT_int.lambda_theory
        state_parameters_section[orb_red_fact_op] = self.get_gL_factor(section_to_cfg)
        
        #state_parameters_section[Ham_red_opt] = JT_int.p_factor
        state_parameters_section[delta_f_opt] = JT_int.delta_f_factor
        state_parameters_section[f_factor_opt] = JT_int.f_factor
        #state_parameters_section[K_JT_opt] = JT_int.KJT_factor

        return state_parameters_section
    
    def save_Taylor_raw_pars_section(self, JT_int:qmp.Exe_tree, section_to_cfg):
        state_parameters_section = {}
        state_parameters_section[int_soc_opt] = self.get_spin_orbit_coupling(section_to_cfg)
        state_parameters_section[orb_red_fact_op] = self.get_gL_factor(section_to_cfg)
        
        state_parameters_section[F_opt] = JT_int.JT_theory.F
        state_parameters_section[G_opt] = JT_int.JT_theory.G

        state_parameters_section[hw_opt] = JT_int.JT_theory.hw_meV

        return state_parameters_section


    def add_magnetic_field_to_cfg(self,new_config:ConfigParser, JT_int:qmp.Exe_tree):


        if self.config.has_section(mag_field):
            new_config.add_section(mag_field)
            new_config[mag_field][min_field_opt] = self.config[mag_field][min_field_opt]
            new_config[mag_field][max_field_opt] = self.config[mag_field][max_field_opt]
            new_config[mag_field][step_num_opt] = self.config[mag_field][step_num_opt]
            new_config[mag_field][dir_vec_opt] = self.config[mag_field][dir_vec_opt]
        return new_config

    def save_raw_pars(self, JT_int:qmp.Exe_tree):
        if self.is_save_raw_pars()==False:
            return
        data_folder = self.get_data_folder_name()
        problem_name = self.get_prefix_name()

        new_config = ConfigParser()

        new_config[single_case_section] = self.save_raw_pars_section(JT_int, single_case_section)
        new_config[essentials_field] = self.config[essentials_field]
        new_config[essentials_field][save_raw_pars_opt] = 'false'

        new_config[atom_structure_field] = JT_int.JT_theory.symm_lattice.create_atom_pars_dict()

        self.add_magnetic_field_to_cfg(new_config, JT_int)



        csv_cfg_name = os.path.join(self.config_file_dir, problem_name+'_csv.cfg')
        with open( csv_cfg_name, 'w') as xml_conf:
            new_config.write(xml_conf)

    def is_save_raw_pars(self):
        if self.config.has_option(essentials_field, save_raw_pars_opt) ==False:
            return False
        else:
            return  True if self.config[essentials_field][save_raw_pars_opt] == 'true' else False

    def conditional_option(self, field_name:str, opt_name:str):
        if self.config.has_option(field_name, opt_name) == False:
            return False
        else:
            return  True if self.config[essentials_field][opt_name] == 'true' else False

    def is_save_model_Hamiltonian_cfg(self):
        return self.conditional_option(essentials_field, save_model_Hamiltonian_cfg_opt)
    def is_save_Taylor_coeffs_cfg(self):
        return self.conditional_option(essentials_field, save_Taylor_coeffs_cfg_opt)


    def is_use_model_hamiltonian(self):
        if self.config.has_option(essentials_field, model_Hamiltonian_opt) ==False:
            return False
        else:
            return  True if self.config[essentials_field][model_Hamiltonian_opt] == 'true' else False
    def is_model_Hamiltonian(self):
        pass

    def get_SOC_split(self,section):
        return self.get_float_option_of_field(section, SOC_split_opt)


    def create_minimal_Exe_tree_from_cfg(self, section_to_look_for):


        energy_split = self.get_SOC_split(section_to_look_for)
        
        gL = self.get_gL_factor(section_to_look_for)
        f_factor = self.get_f_factor(section_to_look_for)
        delta_f = self.get_delta_f(section_to_look_for)
        Yx = self.get_Yx(section_to_look_for)
        Yy = self.get_Yy(section_to_look_for)

        orientation_basis = self.get_basis_col_vectors(essentials_field)

        return qmp.minimal_Exe_tree.from_cfg_data( energy_split,orientation_basis, gL, delta_f,  f_factor, Yx, Yy)

    def get_basis_vector(self,section_id, option_id):
        res:str = self.get_option_of_field(section_id, option_id)
        if res == '':
            return None
        coordinates =  [ float(r) for r  in res.split(',')]

        return V.Vector(*coordinates)

    def get_splitted_strs(self, section_id, option_id, fun):
        res:str = self.get_option_of_field(section_id, option_id)
        if res=='':
            return None
        return [ fun(r) for r  in res.split(',')]

    def get_mag_dir_vector(self) -> maths.col_vector:
        coordinates = self.get_splitted_strs(mag_field, dir_vec_opt, float)
        return maths.col_vector.from_list(coordinates)

    def get_strain_dir_vector(self) -> maths.col_vector:
        coordinates = self.get_splitted_strs(strain_field, strain_vec_op, float)
        return maths.col_vector.from_list(coordinates)

    def get_col_vector(self, field_name, opt_name) ->maths.col_vector:
        coordinates = self.get_splitted_strs(field_name, opt_name, float)
        if coordinates == None:
            return None
        return maths.col_vector.from_list(coordinates)
    
    def get_system_orientation_basis(self):
        return self.get_basis_col_vectors(essentials_field)

    def get_basis_col_vectors(self, field_name) -> list[maths.col_vector]:


        b1 = self.get_col_vector(field_name, basis_vector_1_opt)
        b2 = self.get_col_vector(field_name, basis_vector_2_opt)
        b3 = self.get_col_vector(field_name, basis_vector_3_opt)
        if b1 ==None or b2 == None or b3 == None:
            return None
        else:
            return [ b1.normalize(), b2.normalize(), b3.normalize()  ]

    def get_numbers(self, section_id):
        return [ int(r) for r in self.get_option_of_field(section_id, num_of_atoms_op).split(',')]

    def get_atom_names(self, section_id):
        return self.get_option_of_field(section_id, names_of_atoms_op).split(',')

    def get_masses(self, section_id):
        return [ float(r) for r in self.get_option_of_field(section_id,mass_of_atoms_op).split(',')]

    def get_lattice_energy(self, section_id, energy_opt):
        return float(self.get_option_of_field(section_id, energy_opt))

    def get_F_coeff(self,section_to_look_for):
        return float(self.get_option_of_field(section_to_look_for, F_opt))

    def get_G_coeff(self,section_to_look_for):
        return float(self.get_option_of_field(section_to_look_for, G_opt))
    
    def get_hw(self,section_to_look_for):
        return float(self.get_option_of_field(section_to_look_for, hw_opt))

    def get_p_factor(self, section_to_look_for):
        return self.get_float_option_of_field(section_to_look_for, Ham_red_opt)

    def get_f_factor(self, secton_to_look_for):
        return self.get_float_option_of_field(secton_to_look_for, f_factor_opt)

    def get_delta_p(self,section_to_look_for):
        return float(self.get_option_of_field(section_to_look_for, delta_p_opt))

    def get_delta_f(self,section_to_look_for):
        return float(self.get_option_of_field(section_to_look_for, delta_f_opt))


    def get_Yx(self, section_to_look_for):
        return self.get_float_option_of_field(section_to_look_for, Yx_opt)

    def get_Yy(self, section_to_look_for):
        return self.get_float_option_of_field(section_to_look_for, Yy_opt)

    def get_KJT(self, section_to_look_for):
        return self.get_float_option_of_field(section_to_look_for, K_JT_opt)

    def build_JT_theory_from_Taylor_coeffs(self, section_to_look_for):
        F = self.get_F_coeff( section_to_look_for)
        G = self.get_G_coeff( section_to_look_for)
        hw = self.get_hw( section_to_look_for)
        return jt.Jahn_Teller_Theory().from_Taylor_coeffs(hw,F,G)

    def get_basis_vectors(self, field_name):
        b1_vec = self.get_basis_vector(field_name, basis_vector_1_opt)
        b2_vec = self.get_basis_vector(field_name, basis_vector_2_opt)
        b3_vec = self.get_basis_vector(field_name, basis_vector_3_opt)
        if b1_vec==None or b2_vec==None or b3_vec==None:
            return None
        else:
            return [b1_vec, b2_vec, b3_vec]        

    def build_JT_theory_from_model_Hamiltonian_cfg(self, section_to_look_for):

        p_factor = self.get_p_factor(section_to_look_for)
        lambda_DFT = self.get_spin_orbit_coupling(section_to_look_for)
        KJT = self.get_KJT(section_to_look_for)
        gL = self.get_gL_factor(section_to_look_for)
        #delta_p = self.get_delta_p(section_to_look_for)
        delta_f = self.get_delta_f(section_to_look_for)
        Yx = self.get_Yx(section_to_look_for)
        Yy = self.get_Yy(section_to_look_for)
        f_factor = self.get_f_factor(section_to_look_for)
        return jt.Jahn_Teller_Theory().from_model_parameters( lambda_DFT, KJT, gL, delta_f, Yx, Yy, f_factor)

    def build_jt_theory_from_csv_and_pars_data(self, section_to_look_for):
        data_folder =  self.get_data_folder_name()
        symm_latt_csv_dir = data_folder+ self.get_option_of_field(section_to_look_for, symm_latt_opt)
        JT_latt_csv_dir = data_folder+self.get_option_of_field(section_to_look_for, JT_latt_opt)
        barr_latt_csv_fn = self.get_option_of_field(section_to_look_for, barr_latt_opt)
        barr_latt_csv_dir = data_folder+barr_latt_csv_fn


        basis_vecs = self.get_basis_vectors(atom_structure_field)

        atom_masses = self.get_masses(atom_structure_field)
        atom_numbers = self.get_numbers(atom_structure_field)
        atom_names = self.get_atom_names(atom_structure_field)

        atom_datas = []
        atom_data = namedtuple('atom_data', 'name mass number')


        for (atom_name,atom_mass,atom_number) in zip(atom_names, atom_masses, atom_numbers):
            atom_datas.append( atom_data(atom_name, atom_mass, atom_number))

        sym_lattice_energy = float(self.get_lattice_energy(section_to_look_for,symm_latt_energy_opt))
        less_symm_lattice_1_energy = float(self.get_lattice_energy(section_to_look_for,min_energy_latt_energy_opt))
        if barr_latt_csv_fn!='':
            less_symm_lattice_2_energy = float(self.get_lattice_energy(section_to_look_for,saddle_point_latt_energy_opt))
            less_symm_lattice_2 = V.Lattice().read_from_coordinates_dataframe(barr_latt_csv_dir, atom_datas, basis_vecs,less_symm_lattice_2_energy)
        else:
           less_symm_lattice_2 = None




        symm_lattice = V.Lattice().read_from_coordinates_dataframe(symm_latt_csv_dir, atom_datas,basis_vecs,sym_lattice_energy)
    
        less_symm_lattice_1 = V.Lattice().read_from_coordinates_dataframe(JT_latt_csv_dir, atom_datas, basis_vecs, less_symm_lattice_1_energy)
    
        return jt.Jahn_Teller_Theory(symm_lattice,less_symm_lattice_1,less_symm_lattice_2)#, symm_lattice, less_symm_lattice_1, less_symm_lattice_2

    def is_ZPL_calculation(self):
        return True if self.config.has_section(gnd_state_field) and self.config.has_section(ex_state_field) else False

    def is_from_model_Hamiltonian(self, section_to_look_for:str):
        return True if self.config.has_option(section_to_look_for, Ham_red_opt) or self.config.has_option(section_to_look_for, SOC_split_opt) else False

    def is_from_Taylor_coeffs(self, section_to_look_for:str):
        return True if self.config.has_option(section_to_look_for,F_opt) and self.config.has_option(section_to_look_for,hw_opt) else False

    def is_from_JT_pars(self, section_to_look_for:str):
        return True if self.config.has_option(section_to_look_for,EJT_opt) else False

    def is_from_energy_distance_pairs(self, section_ro_look_for:str):
        return self.config.has_option(section_ro_look_for, con_int_en_opt)

    def is_from_csv(self, section_to_look_for:str):
        high_symm_latt_fn = self.get_option_of_field(section_to_look_for,symm_latt_opt)
        
        len_fn = len(high_symm_latt_fn)

        return True if high_symm_latt_fn[len_fn-4:len_fn] == '.csv' else False

    def is_from_vasprun_xml(self, section_to_look_for:str):
        
        
        high_symm_latt_fn = self.get_option_of_field(section_to_look_for,symm_latt_opt)
        
        len_fn = len(high_symm_latt_fn)

        return True if high_symm_latt_fn[len_fn-4:len_fn] == '.xml' else False


    def read_mag_field(self):
        if self.config.has_section(mag_field):
            self.get_mag_field_dir()
            self.get_mag_field_strengths_list()
        else:
            self.mag_dir_vec = None
            self.mag_field_strengths = None

    def get_mag_field_dir(self):
        dir_str:str = self.get_option_of_field(mag_field,dir_opt )
        
        if dir_str!='':
            dir_list = [ float(x) for x in dir_str.split(',')]
            self.mag_dir_vec = maths.col_vector.from_list(dir_list).normalize()

        else:
            self.mag_dir_vec=None


    def get_field_dir(self, field_name:str):
        dir_str:str = self.get_option_of_field(field_name,dir_opt )
        
        if dir_str!='':
            dir_list = [ float(x) for x in dir_str.split(',')]
            return maths.col_vector.from_list(dir_list).normalize()

        else:
            return None

    def get_B_min(self):
        return float(self.get_option_of_field(mag_field,min_field_opt))

    def get_B_max(self):
        return float(self.get_option_of_field(mag_field,max_field_opt))

    def get_step_num(self):
        return int(self.get_option_of_field(mag_field,step_num_opt))

    def get_field_min(self, field_name:str):
        return float(self.get_option_of_field(field_name,min_field_opt))

    def get_field_max(self, field_name:str):
        return float(self.get_option_of_field(field_name,max_field_opt))


    def get_field_step_num(self, field_name:str):
        return int(self.get_option_of_field(field_name,step_num_opt))
    


    def get_mag_field_strengths_list(self):


        self.B_min = self.get_B_min()
        self.B_max = self.get_B_max()

        self.step_num = self.get_step_num()

        self.mag_field_strengths = np.linspace(self.B_min, self.B_max, self.step_num )

        return self.mag_field_strengths

    def get_field_strengths_list(self):


        min = self.get_field_min()
        max = self.get_field_max()

        step_num = self.get_field_step_num()

        return np.linspace(min, max, step_num )



    def get_res_folder_name(self):
        if self.config.has_option(essentials_field, out_folder_opt):
            folder_path = str(self.config[essentials_field][out_folder_opt])
            return folder_path + '/'
        else:
            return str('')
    

    def get_option_of_field(self, field, option):
        if self.config.has_option( field , option):
            return str(self.config[field][option])
        else:
            return str('')

    def get_float_option_of_field(self, field, option):
        if self.config.has_option( field , option):
            return float(str(self.config[field][option]))
        else:
            return None

    def get_LzSz_exp_val_num(self):
        res_str = self.get_option_of_field(so_c_field,spectrum_range_opt)
        return int(res_str) if res_str!='' else 0

    def get_gL_factor(self):
        res_str = self.get_option_of_field(so_c_field,orb_red_fact_op)
        return int(res_str) if res_str!='' else 0


    def get_data_folder_name(self):
        if self.config.has_option(essentials_field, in_folder_opt):
            folder_path = str(self.config[essentials_field][in_folder_opt])
            return folder_path + '/'
        else:
            return str('')
        

    def get_prefix_name(self):
        if self.config.has_option(essentials_field,out_prefix_opt):
            return str(self.config[essentials_field][out_prefix_opt] )
        else:
            return str('')


    def __init__(self, config_file_path):
        with open(config_file_path,'r') as config_file:
            config_string = config_file.read()
        self.config_file_path:str = config_file_path

        self.config_file_dir = os.path.dirname(config_file_path)

        self.config = ConfigParser()
        self.config.read_string(config_string)
        self.get_defult_values()
    def get_eigen_state_type(self):
        if self.config.has_option(essentials_field, eigen_states_opt):
            return self.get_option_of_field(essentials_field, eigen_states_opt)
        else:
            return 'real'


    def get_defult_values(self):
        self.input_folder = self.get_data_folder_name()
        self.output_folder = self.get_res_folder_name()
        self.max_vib_quant = self.get_order()
        self.output_prefix_name = self.get_prefix_name()
        self.save_raw_parameters = self.is_save_raw_pars()
        self.spectrum_range = self.get_LzSz_exp_val_num()
        self.eigen_states_type = self.get_eigen_state_type()

    def get_spin_orbit_coupling(self, section_to_look_for):
        res_str = self.get_option_of_field(section_to_look_for, int_soc_opt)

        return float( res_str ) if res_str!= '' else 0.0



    def get_gL_factor(self, section_to_look_for = so_c_field):
        return float(self.config[section_to_look_for][orb_red_fact_op] if self.config.has_option(section_to_look_for, orb_red_fact_op) else 0.0)


    def get_order(self):
        return int( self.get_option_of_field(essentials_field, max_vib_quant) )


    def build_JT_theory_new(self, data_folder_name):
        pass


    def build_JT_theories_from_cfg(self, state_JT_field):
        

        if self.config.has_section(state_JT_field):
            E_JT = float(self.get_option_of_field(state_JT_field,EJT_opt ))
            
            delta_field = self.get_option_of_field(state_JT_field, E_barr_opt)
            delta_meV =float(delta_field) if delta_field!='' else 0.0
            hw = float(self.get_option_of_field(state_JT_field, hw_opt))

            JT_theory = jt.Jahn_Teller_Theory()
            
            JT_theory.E_JT_meV = E_JT
            JT_theory.delta_meV = delta_meV
            JT_theory.hw_meV = hw
            JT_theory.order_flag = 2
            
            
            JT_theory.JT_dist = float(self.get_option_of_field(state_JT_field, symm_min_dist_opt))
            JT_theory.barrier_dist = float(self.get_option_of_field(state_JT_field, symm_saddl_dist_opt))
            
            JT_theory.calc_paramters_until_second_order_from_JT_pars()

            JT_theory.orbital_red_fact = self.get_gL_factor(state_JT_field)
            JT_theory.intrinsic_soc = self.get_spin_orbit_coupling(state_JT_field)

            return JT_theory

    def build_JT_theories_from_energy_distance_pairs(self, field_to_look_for):
        con_int_en = self.get_float_option_of_field(field_to_look_for,con_int_en_opt)
        con_int_loc = self.get_float_option_of_field(field_to_look_for,con_int_loc_opt)

        saddle_point_en = self.get_float_option_of_field(field_to_look_for,saddle_point_latt_energy_opt)
        saddle_point_loc = self.get_float_option_of_field(field_to_look_for, saddle_point_loc_opt)

        minimum_en = self.get_float_option_of_field(field_to_look_for,min_energy_latt_energy_opt)
        minimum_loc = self.get_float_option_of_field(field_to_look_for, global_min_loc_opt)

        JT_en = abs(con_int_en-minimum_en)
        barr_en = abs(saddle_point_en-minimum_en)

        JT_dist = abs(con_int_loc-minimum_loc)
        barr_dist = abs(con_int_loc - saddle_point_loc)

        JT_theory = jt.Jahn_Teller_Theory()

        JT_theory.E_JT_meV = JT_en
        JT_theory.JT_dist = JT_dist
        JT_theory.delta_meV = barr_en
        JT_theory.barrier_dist = barr_dist

        JT_theory.order_flag = 2

        JT_theory.calc_hw()





        JT_theory.calc_Taylor_coeffs_K()



        JT_theory.orbital_red_fact = self.get_gL_factor(field_to_look_for)
        JT_theory.intrinsic_soc = self.get_spin_orbit_coupling(field_to_look_for)
        
        return JT_theory