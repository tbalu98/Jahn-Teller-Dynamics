from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jahn_teller_dynamics.math.maths as maths
from jahn_teller_dynamics.math.maths import exp_func, fit_function

plt.figure(figsize=(6, 6))  # Width = 10 inches, height = 6 inches

plt.rcParams['font.size'] = 16


folder = 'results/N3V_HSE/'
folder = 'results/N3V_PBE/'


df = pd.read_csv(folder + 'N3V_SOC_PBE.csv')

lattice_cnst = df['lattice_cnst']
gamma_SOC = df['gamma'].to_numpy()*1000
half_SOC = df['half'].to_numpy()*1000

# Get fit parameters for gamma_SOC
p0 = [0.01, 0.5, 1.5]  # Initial guesses for a, b, c

a_gamma, b_gamma, c_gamma = fit_function(lattice_cnst, gamma_SOC,p0,exp_func)[0]
print(a_gamma, b_gamma, c_gamma)
#a_half, b_half, c_half = fit_exponential(lattice_cnst, half_SOC)[0]

x_from = min(lattice_cnst)
x_to = 3*max(lattice_cnst)
# Generate smooth curve for plotting
x_fit = np.linspace(x_from, x_to, 100)
y_fit_gamma = exp_func(x_fit, a_gamma, b_gamma, c_gamma)
#y_fit_half = exp_func(x_fit, a_half, b_half, c_half)
plt.plot(lattice_cnst, gamma_SOC, 'ro', label = 'DFT calculation')
#plt.plot(lattice_cnst, half_SOC, 'bo', label = 'DFT calculation')



plt.plot(x_fit, y_fit_gamma, 'r-', label='exponential fit')
#plt.plot(x_fit, y_fit_half, 'b-', label='Exponential fit')  


#Convergence line
plt.plot(x_fit, x_fit*0 + c_gamma, 'k--', label = 'convergence line')

#plt.plot(lattice_cnst, half_SOC, 'bo')
plt.title('Spin-orbit coupling in the ' + r'$\Gamma$ point')

plt.xlabel('supercell size (Å)')
plt.ylabel('spin-orbit coupling (meV)')
plt.legend()
plt.show()
