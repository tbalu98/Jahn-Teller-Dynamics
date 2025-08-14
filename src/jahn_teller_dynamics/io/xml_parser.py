import xml.dom.minidom
import jahn_teller_dynamics.io.VASP as V
from configparser import ConfigParser
import jahn_teller_dynamics.math.maths as maths



def save_raw_data_from_xmls(lattices:list[V.Lattice], problem_name, data_folder, xml_cfg:ConfigParser):
    



    symm_lattice = lattices[0]



    xml_cfg.remove_section('vasprun.xml_files')
    xml_cfg.add_section('.csv_files')
    
    csv_files = {}


    df =  symm_lattice.to_coordinates_data_frame()
    symm_latt_geom_filename = problem_name+'_symmetric_lattice.csv'
    df.to_csv(data_folder +   symm_latt_geom_filename)
    csv_files['symmetric_lattice'] = symm_latt_geom_filename

    


    less_symm_lattice_1 = lattices[1]
    less_symm_latt_geom_filename_1 = problem_name+'_JT_lattice.csv'
    df =  less_symm_lattice_1.to_coordinates_data_frame()
    df.to_csv(data_folder + less_symm_latt_geom_filename_1)
    csv_files['Jahn-Teller_lattice'] = less_symm_latt_geom_filename_1


    if lattices[-1]!=None:
        less_symm_lattice_2 = lattices[2]

        df =  less_symm_lattice_2.to_coordinates_data_frame()
        barrier_lattice_filename = problem_name + '_barrier_lattice.csv'
        df.to_csv( data_folder + barrier_lattice_filename)
        csv_files['barrier_lattice'] = barrier_lattice_filename

    else:
        less_symm_lattice_2=None


    par_cfg = ConfigParser()

    atom_names_dict = {}
    atom_mass_dict = {}
    atom_numbers_dict = {}
    for i,ions_arr in zip(range(1,len(symm_lattice.ions_arr)+1) ,symm_lattice.ions_arr):
        atom_names_dict[ 'atom_'+str(i) + '_name'  ] = ions_arr.name
        atom_mass_dict[ 'atom_'+str(i) + '_mass'  ] = ions_arr.m
        atom_numbers_dict['atom_' + str(i) + '_number'] = len(ions_arr)
        





    par_cfg['atom_names'] = atom_names_dict
    par_cfg['atom_masses'] = atom_mass_dict
    par_cfg['atom_numbers'] = atom_numbers_dict
    par_cfg['lattice_energies'] = {'symm_lattice_energy':symm_lattice.energy, 'JT_lattice_energy': less_symm_lattice_1.energy, 'barrier_lattice_energy': less_symm_lattice_2.energy }

    par_cfg['basis_vectors'] = { 
    'basis_vector_1_x' :symm_lattice.ions_arr[0].basis_vecs[0].x,
    'basis_vector_1_y' : symm_lattice.ions_arr[0].basis_vecs[0].y,
    'basis_vector_1_z' : symm_lattice.ions_arr[0].basis_vecs[0].z,

    'basis_vector_2_x' : symm_lattice.ions_arr[0].basis_vecs[1].x,
    'basis_vector_2_y' : symm_lattice.ions_arr[0].basis_vecs[1].y,
    'basis_vector_2_z' : symm_lattice.ions_arr[0].basis_vecs[1].z,

    'basis_vector_3_x' : symm_lattice.ions_arr[0].basis_vecs[2].x,
    'basis_vector_3_y' : symm_lattice.ions_arr[0].basis_vecs[2].y,
    'basis_vector_3_z' : symm_lattice.ions_arr[0].basis_vecs[2].z}


    at_pars_filename  = problem_name + '_atom_parameters.cfg'
    csv_files['atom_parameters'] = at_pars_filename
    with open( data_folder +  problem_name+'_atom_parameters.cfg', 'w') as conf:
        par_cfg.write(conf)

    xml_cfg['.csv_files'] = csv_files

    with open( data_folder +  problem_name+'_csv.cfg', 'w') as xml_conf:
        xml_cfg.write(xml_conf)



class xml_parser:
    def __init__(self, xml_file_name:str):
        print('loading ' + xml_file_name)
        self.dom_tree = xml.dom.minidom.parse(xml_file_name)
        self.group:xml.dom.minidom.Element = self.dom_tree.documentElement
        self.get_last_calculation()
        self.get_last_scstep()
        self.get_structure()
        self.get_crystal_arrs()
        self.get_basis_vecs()
        self.get_pos_vecs()
        self.get_atom_info()
        self.get_lattice_energy()
        self.build_lattice()



    def build_lattice(self):
        
        position_vecs_iterator = iter(self.pos_vecs)


        ions_arr = []
        for atom_type_attr in self.atom_type_attrs:

            ion_num = atom_type_attr[0]
            ion_name = atom_type_attr[1]
            ion_mass = atom_type_attr[2]

            ion_pos_vecs = []

            for i in range(0,ion_num):
                ion_pos_vecs.append(next(position_vecs_iterator))
            
            ions_arr.append(V.Ions( name=ion_name, vecs=ion_pos_vecs, m = ion_mass, cell_x = self.basis_vecs[0].x,cell_y = self.basis_vecs[0].y,cell_z = self.basis_vecs[0].z , basis_vecs=self.basis_vecs ))
        basis_col_vecs = [ maths.col_vector.from_list(basis_vec.tolist() ) for basis_vec in self.basis_vecs ]
        self.lattice = V.Lattice(self.lattice_energy,basis_col_vecs)
        self.lattice.ions_arr = ions_arr





    def get_pos_vecs(self):

        strc_arrs = self.structure.getElementsByTagName('varray')
        positions = [ strc_arr for strc_arr in strc_arrs if strc_arr.getAttribute('name')=='positions' ][-1]
        self.pos_vecs = xml_parser.get_vectors_from_varray(positions)


    def get_structure(self):
        self.structure = [ x for x in self.group.getElementsByTagName('structure') if x.getAttribute('name')=='finalpos'][-1]

    def get_crystal_arrs(self):
        self.crystal = self.structure.getElementsByTagName('crystal')[-1]
        self.cristal_arrs = self.crystal.getElementsByTagName('varray')
        

    def get_basis_vecs(self):

        basis = [ cristal_arr for cristal_arr in self.cristal_arrs if cristal_arr.getAttribute('name')=='basis' ][-1]
        
        self.basis_vecs = xml_parser.get_vectors_from_varray(basis)


    def get_vectors_from_varray(varray:xml.dom.minidom.Element):
        vectors = varray.getElementsByTagName('v')
        return [ V.Vector.from_str(v.childNodes[0].data)   for v in vectors ]


        

    def get_lattice_energy(self):
        energy = self.last_scstep.getElementsByTagName('energy')[-1]
        energy_attrs = energy.getElementsByTagName('i')
        self.lattice_energy = float([ energy_attr.childNodes[0].data for energy_attr in energy_attrs if energy_attr.getAttribute('name')=='e_fr_energy' ][-1])


    def get_last_scstep(self):
        self.last_scstep = self.last_calculation.getElementsByTagName('scstep')[-1]

    def get_last_calculation(self):
        self.last_calculation = self.group.getElementsByTagName('calculation')[-1]



    def get_atom_info(self):
        atominfo = self.group.getElementsByTagName('atominfo')[-1]

        arrays = atominfo.getElementsByTagName('array')

        atomtypes = [ array for array in arrays if array.getAttribute('name') == 'atomtypes'  ][-1].getElementsByTagName('set')[-1].getElementsByTagName('rc')

        self.atom_type_attrs = []

        for atomtype in atomtypes:
            attrs = [ attr.childNodes[0].data for attr in atomtype.getElementsByTagName('c')[0:3]]
            self.atom_type_attrs.append([ int(attrs[0]), attrs[1].replace(' ', ''), float(attrs[2]) ])

        