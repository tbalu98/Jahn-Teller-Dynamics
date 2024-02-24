import xml.dom.minidom
import utilities.VASP as VASP


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

        