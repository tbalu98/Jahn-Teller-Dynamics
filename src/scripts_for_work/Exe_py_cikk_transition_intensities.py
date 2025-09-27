import numpy as np
import matplotlib.pyplot as plt

# Generate mock data (replace with your data)
def plot_zpl_intensity_scatter(magnetic_field, energy, intensity):
    """
    Plot ZPL intensity as a function of magnetic field and energy
    
    Parameters:
    -----------
    magnetic_field : np.ndarray
        Magnetic field values in Tesla
    energy : np.ndarray 
        Energy values in eV
    intensity : np.ndarray
        2D array of intensity values with shape (len(magnetic_field), len(energy))
    """
    # Create a grid for scatter plot
    B_grid, E_grid = np.meshgrid(magnetic_field, energy)
    B_flat = B_grid.ravel()
    E_flat = E_grid.ravel()
    I_flat = intensity.T.ravel()  # Transpose to match meshgrid

    # Plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        B_flat, E_flat, c=I_flat, 
        s=10,  # Marker size
        cmap='viridis', 
        edgecolor='k', 
        linewidth=0.0
    )

    # Labels and colorbar
    plt.xlabel("Magnetic Field (T)")
    plt.ylabel("Energy (eV)")
    plt.title("ZPL Intensity vs. Magnetic Field and Energy")
    cbar = plt.colorbar(sc, label="Intensity (a.u.)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Discrete_Contour_Plot.pdf", dpi=300)

magnetic_field = np.array([0, 1.2, 2.5, 3.1, 4.3])  # Tesla
energy = np.array([.5, 1.0, 1.1, 1.2])     # eV
intensity = np.random.rand(len(magnetic_field), len(energy))  # Random intensities (replace with your data)

plot_zpl_intensity_scatter(magnetic_field, energy, intensity)


plt.show()