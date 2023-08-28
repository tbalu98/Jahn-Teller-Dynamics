import utilities.VASP as vasp
import math
import utilities.quantum_mechanics as qm
gnd_211_data = vasp.POSCAR_data('SnV/Gnd_211.geom')
gnd_D3d_data = vasp.POSCAR_data('SnV/Gnd_D3d.geom')
gnd_inner_data = vasp.POSCAR_data('SnV/Gnd_inner.geom')

gnd_D3d_lattice = gnd_D3d_data.lattice
gnd_211_lattice = gnd_211_data.lattice

m_Ge = 118.71 
m_C = 12.01
lattice_energy_D3d = -5368.3068
lattice_energy_inner = -5368.3268
lattice_energy_211 = -5368.3284


gnd_211_lattice.get_ions('Ge').m = m_Ge
gnd_211_lattice.get_ions('C').m = m_C
gnd_211_lattice.energy = lattice_energy_211



gnd_D3d_lattice.get_ions('Ge').m = m_Ge
gnd_D3d_lattice.get_ions('C').m = m_C
gnd_D3d_lattice.energy = lattice_energy_D3d



dist_E_JT =  gnd_D3d_lattice.calc_dist(gnd_211_lattice)

print(dist_E_JT)