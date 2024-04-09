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

    def get_masses(self):
        return float(self.config['atom_masses']['atom_1_mass']), float(self.config['atom_masses']['atom_2_mass'])

    def get_names(self):
        return self.config['atom_names']['atom_1_name'], self.config['atom_names']['atom_2_name']

    def get_lattice_energy(self, lattice_name):
        return float(self.config['lattice_energies'][lattice_name])

class Jahn_Teller_config_parser:

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
    def __init__(self, config_file_name):
        config_file = open(config_file_name,'r')
        config_string = config_file.read()

        self.config = ConfigParser()
        self.config.read_string(config_string)

    def get_spin_orbit_coupling(self):
        return float(self.config['spin_orbit_coupling']['lambda'] if self.config.has_section('spin_orbit_coupling') else 0.0)

    def get_order(self):
        return int(self.config['DEFAULT']['order'] if self.config.has_option('DEFAULT', 'order') else 12)

    def build_JT_theory(self):
        filenames = []
        if self.config.has_section('vasprun.xml_files'):
            if self.config.has_option('vasprun.xml_files','symmetric_lattice'):
                filenames.append(self.config['vasprun.xml_files']['symmetric_lattice'])

            if self.config.has_option('vasprun.xml_files','Jahn-Teller_lattice'):
                filenames.append(self.config['vasprun.xml_files']['Jahn-Teller_lattice'])
            
            if self.config.has_option('vasprun.xml_files','barrier_lattice'):
                filenames.append(self.config['vasprun.xml_files']['barrier_lattice'])
        
            return jt.Jahn_Teller_Theory.build_jt_theory_from_vasprunxmls(filenames)

        elif self.config.has_section('.csv_files'):
            if self.config.has_option('.csv_files', 'atom_parameters'):
                filenames.append(self.config['.csv_files']['atom_parameters'])

            if self.config.has_option('.csv_files','symmetric_lattice'):
                filenames.append(self.config['.csv_files']['symmetric_lattice'])

            if self.config.has_option('.csv_files','Jahn-Teller_lattice'):
                filenames.append(self.config['.csv_files']['Jahn-Teller_lattice'])
            
            if self.config.has_option('.csv_files','barrier_lattice'):
                filenames.append(self.config['.csv_files']['barrier_lattice'])

            return jt.build_jt_theory_from_csv_2(filenames)
        
        elif self.config.has_section('fitting_parameters'):
            F = float(self.config['fitting_parameters']['F'])
            G = float(self.config['fitting_parameters']['G'])
            hw = float(self.config['fitting_parameters']['hw'])


            jt_theory = jt.Jahn_Teller_Theory()
            return jt_theory.from_Taylor_coeffs(hw, F, G), None, None,None

