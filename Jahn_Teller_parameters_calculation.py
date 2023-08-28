import utilities.VASP as vasp
import math
import utilities.quantum_mechanics as qm
gnd_211_data = vasp.POSCAR_data('SnV/Gnd_211.geom')
gnd_D3d_data = vasp.POSCAR_data('SnV/Gnd_D3d.geom')
gnd_inner_data = vasp.POSCAR_data('SnV/Gnd_inner.geom')

gnd_D3d_lattice = gnd_D3d_data.lattice
gnd_211_lattice = gnd_211_data.lattice
gnd_inner_lattice = gnd_inner_data.lattice

m_Sn = 118.71 
m_C = 12.01
lattice_energy_D3d = -5368.3068
lattice_energy_inner = -5368.3268
lattice_energy_211 = -5368.3284


gnd_211_lattice.get_ions('Ge').m = m_Sn
gnd_211_lattice.get_ions('C').m = m_C
gnd_211_lattice.energy = lattice_energy_211



gnd_D3d_lattice.get_ions('Ge').m = m_Sn
gnd_D3d_lattice.get_ions('C').m = m_C
gnd_D3d_lattice.energy = lattice_energy_D3d

gnd_inner_lattice.get_ions('Ge').m = m_Sn
gnd_inner_lattice.get_ions('C').m = m_C
gnd_inner_lattice.energy = lattice_energy_inner


dist_barrier =  gnd_D3d_lattice.calc_dist(gnd_211_lattice)
print(dist_barrier)
dist_E_JT = gnd_D3d_lattice.calc_dist(gnd_inner_lattice)
print(dist_E_JT)

#JT = qm.Jahn_Teller_Theory(lattice_energy_inner,lattice_energy_211, lattice_energy_D3d,D3d_inner_dist,D3d_211_dist )


symm_latt_energy = lattice_energy_D3d
barrier_latt_energy = lattice_energy_inner
E_JT_latt_energy = lattice_energy_211

"""
symm_latt_en = -5361.9405
barrier_latt_energy = -5362.0198
E_JT_latt_energy = -5362.0321

dist_barrier = 0.34363
dist_E_JT = 0.33619

"""

JT = qm.Jahn_Teller_Theory( barrier_latt_energy, E_JT_latt_energy, symm_latt_energy,dist_barrier, dist_E_JT ) 
print(JT)

