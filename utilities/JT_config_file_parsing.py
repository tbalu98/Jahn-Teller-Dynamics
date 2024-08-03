from configparser import ConfigParser
import utilities.jahn_teller_theory as jt
import utilities.VASP as VASP

class Atom_config_parser:
    def __init__(self, config_file_name):
        config_file = open(config_file_name,'r')
        config_string = config_file.read()

        self.config = ConfigParser()
        self.config.read_string(config_string)        

    def get_basis_vectors(self):
        bv_1x = float(self.config['basis_vectors']['basis_vector_1_x'])
        bv_1y = float(self.config['basis_vectors']['basis_vector_1_y'])
        bv_1z = float(self.config['basis_vectors']['basis_vector_1_z'])
        
        bv_2x = float(self.config['basis_vectors']['basis_vector_2_x'])
        bv_2y = float(self.config['basis_vectors']['basis_vector_2_y'])
        bv_2z = float(self.config['basis_vectors']['basis_vector_2_z'])

        bv_3x = float(self.config['basis_vectors']['basis_vector_3_x'])
        bv_3y = float(self.config['basis_vectors']['basis_vector_3_y'])
        bv_3z = float(self.config['basis_vectors']['basis_vector_3_z'])

        return VASP.Vector(bv_1x, bv_1y, bv_1z), VASP.Vector(bv_2x, bv_2y, bv_2z),VASP.Vector(bv_3x, bv_3y, bv_3z)


    def get_numbers(self):
        numbers = [   ]
        
        for i in range(1, len(self.config['atom_numbers']) + 1):
            numbers.append(self.config['atom_numbers'][ 'atom_' + str(i)+ '_number' ])

        return numbers

    def get_masses(self):
        masses = [   ]
        
        for i in range(1, len(self.config['atom_masses']) + 1):
            masses.append(self.config['atom_masses'][ 'atom_' + str(i)+ '_mass' ])

        return masses

        #return float(self.config['atom_masses']['atom_1_mass']), float(self.config['atom_masses']['atom_2_mass'])

    def get_names(self):
        names = [   ]
        
        for i in range(1, len(self.config['atom_names']) + 1):
            names.append(self.config['atom_names'][ 'atom_' + str(i)+ '_name' ])

        return names


        #return self.config['atom_names']['atom_1_name'], self.config['atom_names']['atom_2_name']

    def get_lattice_energy(self, lattice_name):
        return float(self.config['lattice_energies'][lattice_name])

class ZPL_config_parser:
    def __init__(self, config_file_name):
        config_file = open(config_file_name,'r')
        config_string = config_file.read()

        self.config = ConfigParser()
        self.config.read_string(config_string)

    def get_option_of_field(self, field, option):
        if self.config.has_option( field , option):
            return str(self.config[field][option])
        else:
            return str('')
    
    def get_calculation_name(self):
        return self.get_option_of_field('DEFAULT', 'calculation_name')

    def get_gnd_cfg_filename(self):
        return self.get_option_of_field('DEFAULT', 'ground_state_cfg')
    def get_cfg_data_folder(self):
        return self.get_option_of_field('DEFAULT', 'cfg_data_folder') + '/'

    def get_results_folder(self):
        return self.get_option_of_field('DEFAULT', 'results_folder') + '/'

    def get_ex_cfg_filename(self):
        return self.get_option_of_field('DEFAULT', 'excited_state_cfg')
    
    def get_B_min(self):
        return float(self.get_option_of_field('magnetic_field', 'B_min'))

    def get_B_max(self):
        return float(self.get_option_of_field('magnetic_field', 'B_max'))

    def get_step_num(self):
        return int(self.get_option_of_field('magnetic_field', 'step_num'))


class Jahn_Teller_config_parser:

    def get_res_folder_name(self):
        if self.config.has_option('DEFAULT', 'results_folder'):
            return str(self.config['DEFAULT']['results_folder']) + '/'
        else:
            return str('')
    

    def get_option_of_field(self, field, option):
        if self.config.has_option( field , option):
            return str(self.config[field][option])
        else:
            return str('')

    def get_LzSz_exp_val_num(self):
        res_str = self.get_option_of_field('spin_orbit_coupling','calc_LzSz')
        return int(res_str) if res_str!='' else 0

    def get_gL_factor(self):
        res_str = self.get_option_of_field('spin_orbit_coupling','gL')
        return int(res_str) if res_str!='' else 0


    def get_data_folder_name(self):
        if self.config.has_option('DEFAULT', 'data_folder'):
            return str(self.config['DEFAULT']['data_folder']) + '/'
        else:
            return str('')
        

    def get_problem_name(self):
        if self.config.has_option('DEFAULT','calculation_name'):
            return str(self.config['DEFAULT']['calculation_name'] )
        else:
            return str('')

    def get_electric_field(self):
        if self.config.has_section('electric_field'):
            return float(self.config['electric_field']['E_x'] if self.config.has_option('electric_field', 'E_x') else 0.0), float(self.config['electric_field']['E_y'] if self.config.has_option('electric_field', 'E_y') else 0.0)
        else:
            return 0.0,0.0
    
    def get_magnetic_field(self):
        if self.config.has_section('magnetic_field'):
            return float(self.config['magnetic_field']['B_x'] if self.config.has_option('magnetic_field', 'B_x') else 0.0),float(self.config['magnetic_field']['B_y'] if self.config.has_option('magnetic_field', 'B_y') else 0.0), float(self.config['magnetic_field']['B_z'] if self.config.has_option('magnetic_field', 'B_z') else 0.0)
        else:
            return 0.0,0.0, 0.0
    """
    def get_magnetic_field(self):
        if self.config.has_section('magnetic_field'):
            return float(self.config['magnetic_field']['B_z'] if self.config.has_option('magnetic_field', 'B_z') else 0.0)
    """
    def __init__(self, config_file_name):
        config_file = open(config_file_name,'r')
        config_string = config_file.read()

        self.config = ConfigParser()
        self.config.read_string(config_string)

    def get_spin_orbit_coupling(self):
        return float(self.config['spin_orbit_coupling']['lambda'] if self.config.has_section('spin_orbit_coupling') else 0.0)

    def get_gL_factor(self):
        return float(self.config['spin_orbit_coupling']['gL'] if self.config.has_section('spin_orbit_coupling') else 0.0)


    def get_order(self):
        return int(self.config['DEFAULT']['order'] if self.config.has_option('DEFAULT', 'order') else 12)

    def build_JT_theory(self, data_folder_name):
        filenames = []
        if self.config.has_section('vasprun.xml_files'):
            if self.config.has_option('vasprun.xml_files','symmetric_lattice'):
                filenames.append(data_folder_name + self.config['vasprun.xml_files']['symmetric_lattice'])

            if self.config.has_option('vasprun.xml_files','Jahn-Teller_lattice'):
                filenames.append(data_folder_name + self.config['vasprun.xml_files']['Jahn-Teller_lattice'])
            
            if self.config.has_option('vasprun.xml_files','barrier_lattice'):
                filenames.append(data_folder_name + self.config['vasprun.xml_files']['barrier_lattice'])
        
            return jt.Jahn_Teller_Theory.build_jt_theory_from_vasprunxmls(filenames)

        elif self.config.has_section('.csv_files'):
            if self.config.has_option('.csv_files', 'atom_parameters'):
                filenames.append(data_folder_name + self.config['.csv_files']['atom_parameters'])

            if self.config.has_option('.csv_files','symmetric_lattice'):
                filenames.append(data_folder_name + self.config['.csv_files']['symmetric_lattice'])

            if self.config.has_option('.csv_files','Jahn-Teller_lattice'):
                filenames.append(data_folder_name + self.config['.csv_files']['Jahn-Teller_lattice'])
            
            if self.config.has_option('.csv_files','barrier_lattice'):
                filenames.append(data_folder_name + self.config['.csv_files']['barrier_lattice'])

            return jt.Jahn_Teller_Theory.build_jt_theory_from_csv(filenames)
        
        elif self.config.has_section('fitting_parameters'):
            F = float(self.config['fitting_parameters']['F'])
            G = float(self.config['fitting_parameters']['G'])
            hw = float(self.config['fitting_parameters']['hw'])


            jt_theory = jt.Jahn_Teller_Theory()
            return jt_theory.from_Taylor_coeffs(hw, F, G), None, None,None

