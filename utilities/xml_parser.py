import xml.dom.minidom
import utilities.VASP as VASP
import pandas as pd
from configparser import ConfigParser

def save_raw_data_from_xmls(lattices:list[VASP.Lattice], problem_name, data_folder):
    



    symm_lattice = lattices[0]


#symm_lattice = VASP.Lattice().read_from_coordinates_dataframe('symm_latt.csv',{'Sn': 16, 'C': 12})

    df =  symm_lattice.to_coordinates_data_frame()
    df.to_csv(data_folder +   problem_name+'_symmetric_lattice.csv')



    less_symm_lattice_1 = lattices[1]

    df =  less_symm_lattice_1.to_coordinates_data_frame()
    df.to_csv(data_folder + problem_name+'_JT_lattice.csv')


    if lattices[-1]!=None:
        less_symm_lattice_2 = lattices[2]

        df =  less_symm_lattice_2.to_coordinates_data_frame()
        df.to_csv( data_folder + problem_name + '_barrier_lattice.csv')
    else:
        less_symm_lattice_2=None

    """
    par_dict = {}

    par_dict['atom_1_name'] = [symm_lattice.ions_arr[0].name]
    par_dict['atom_2_name'] = [symm_lattice.ions_arr[1].name]

    par_dict['atom_1_mass'] = [symm_lattice.ions_arr[0].m]
    par_dict['atom_2_mass'] = [symm_lattice.ions_arr[1].m]

    par_dict['symm_lattice_energy'] = [symm_lattice.energy]
    par_dict['JT_lattice_energy'] = [less_symm_lattice_1.energy]
    if less_symm_lattice_2!=None:

        par_dict['barrier_lattice_energy'] = [less_symm_lattice_2.energy]

    par_dict['basis_vec_1_x'] = [symm_lattice.ions_arr[0].basis_vecs[0].x]
    par_dict['basis_vec_1_y'] = [symm_lattice.ions_arr[0].basis_vecs[0].y]
    par_dict['basis_vec_1_z'] = [symm_lattice.ions_arr[0].basis_vecs[0].z]

    par_dict['basis_vec_2_x'] = [symm_lattice.ions_arr[0].basis_vecs[1].x]
    par_dict['basis_vec_2_y'] = [symm_lattice.ions_arr[0].basis_vecs[1].y]
    par_dict['basis_vec_2_z'] = [symm_lattice.ions_arr[0].basis_vecs[1].z]

    par_dict['basis_vec_3_x'] = [symm_lattice.ions_arr[0].basis_vecs[2].x]
    par_dict['basis_vec_3_y'] = [symm_lattice.ions_arr[0].basis_vecs[2].y]
    par_dict['basis_vec_3_z'] = [symm_lattice.ions_arr[0].basis_vecs[2].z]


    par_df = pd.DataFrame(par_dict)

    par_df.index.name = 'index'


    par_df.to_csv(data_folder + problem_name +'_atomic_parameters.csv', sep = ';')
    """

    par_cfg = ConfigParser()

    atom_names_dict = {}
    atom_mass_dict = {}

    for i,ions_arr in zip(range(1,len(symm_lattice.ions_arr)+1) ,symm_lattice.ions_arr):
        atom_names_dict[ 'atom_'+str(i) + '_name'  ] = ions_arr.name
        atom_mass_dict[ 'atom_'+str(i) + '_mass'  ] = ions_arr.m
        




    #par_cfg['atom_names'] = { 'atom_1_name' : symm_lattice.ions_arr[0].name, 'atom_2_name':symm_lattice.ions_arr[1].name }
    #par_cfg['atom_masses'] = { 'atom_1_mass':symm_lattice.ions_arr[0].m, 'atom_2_mass':symm_lattice.ions_arr[1].m}
    par_cfg['atom_names'] = atom_names_dict
    par_cfg['atom_masses'] = atom_mass_dict
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

    with open( data_folder +  problem_name+'_atom_parameters.cfg', 'w') as conf:
        par_cfg.write(conf)




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
            
            ions_arr.append(VASP.Ions( name=ion_name, vecs=ion_pos_vecs, m = ion_mass, cell_x = self.basis_vecs[0].x,cell_y = self.basis_vecs[0].y,cell_z = self.basis_vecs[0].z , basis_vecs=self.basis_vecs ))
        
        self.lattice = VASP.Lattice(energy = self.lattice_energy,cell_x=self.basis_vecs[0].x,cell_y=self.basis_vecs[1].y,cell_z=self.basis_vecs[2].z)
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
        return [ VASP.Vector.from_str(v.childNodes[0].data)   for v in vectors ]


        

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

        