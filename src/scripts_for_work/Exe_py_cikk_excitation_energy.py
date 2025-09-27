import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from jahn_teller_dynamics.math import exp_func, fit_function
def exponential_fit(x, c, k):
    return c - np.exp(k * x)


folder_name = 'results/N3V_PBE/'
filename = 'N3V_excitation_energy.csv'
# read the data from the csv file
data = pd.read_csv(folder_name + filename)

number_of_atoms = data['number_of_atoms'].values
excitation_energy = data['excitation_energy'].values


#p0 = [2.4, 1]
#popt, pcov = fit_function(number_of_atoms, excitation_energy, p0, exponential_fit)

# Generate points for smooth curve
x_fit = np.linspace(min(number_of_atoms), max(number_of_atoms), 100)
#y_fit = exponential_fit(x_fit, *popt)

# Print fit parameters
#print(f"Fit parameters: c = {popt[0]:.3f}, k = {popt[1]:.3f}")

#plt.plot(x_fit, y_fit, 'r-')


plt.rcParams['font.size'] = 15
plt.title('Excitation energy of N3V')
plt.plot(number_of_atoms, excitation_energy, 'o')
plt.xlabel('number of atoms')
plt.ylabel('excitation energy (eV)')



plt.show()



