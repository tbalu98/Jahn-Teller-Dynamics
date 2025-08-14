import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm

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
        magnetic_field, energy, c=intensity, 
        s=10,  # Marker size
        cmap='jet', 
        edgecolor='k', 
        linewidth=0.0,
        
    )

    # Labels and colorbar
    plt.xlabel("Magnetic Field (T)")
    plt.ylabel("Energy (eV)")
    plt.title("ZPL Intensity vs. Magnetic Field and Energy")
    cbar = plt.colorbar(sc, label="Intensity (a.u.)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Discrete_Contour_Plot.pdf", dpi=300)

class contour_data():

    def get_dim(self):
        return len(self.data_df.index.to_list())*len(self.data_df.columns.to_list())
    
    def from_df(self, df):
        self.data_df = df
        return self

    def from_raw_data(self , data_mx, index, columns):
        self.data_df = pd.DataFrame(data_mx, index = index, columns = columns)
        return self

    def from_csv(self, path):
        self.data_df = pd.read_csv(path, index_col= 'intensity')
        return self
        

    def save_contour(self, path):
        self.data_df.to_csv(path)

    def index_trf(self, trf_fun, label ):
        or_indexes = self.data_df.index.to_list()
        trf_indexes = [ trf_fun(x) for x in or_indexes]
        self.data_df[label] = trf_indexes
        self.data_df = self.data_df.set_index(label)

    def create_contour(self):
    
        y_valsor = np.array(self.data_df.index.to_list())


        cols = self.data_df.columns.to_list()
        cols = list(map(complex, cols))
        x_valsor = np.array(list(map( lambda x: float(x.real) ,cols)))

        x_matrix, y_matrix = np.meshgrid( x_valsor,y_valsor)

        data_mx = self.data_df.to_numpy(dtype = np.complex64)

        max_of_contour =  np.max(self.data_df.to_numpy(dtype=np.complex64).flatten())
        min_of_contour =  np.min(self.data_df.to_numpy(dtype=np.complex64).flatten())

        levels = np.linspace(min_of_contour, max_of_contour, 10000)

        plt.xlim(x_valsor[0],x_valsor[-1])
        plt.ylim(y_valsor[0],y_valsor[-1])
        
        

        cs = plt.contourf(x_matrix, y_matrix , data_mx,  cmap='jet', levels = levels )
        

        cs.changed()
        return cs

    def max(self):
        return np.max(self.data_df.to_numpy(dtype=np.complex64).flatten())

    def min(self):
        return np.min(self.data_df.to_numpy(dtype=np.complex64).flatten())

    def create_contour_ax(self, axis, levels):
        
        y_valsor = np.array(self.data_df.index.to_list())


        cols = self.data_df.columns.to_list()
        cols = list(map(complex, cols))
        x_valsor = np.array(list(map( lambda x: float(x.real) ,cols)))

        x_matrix, y_matrix = np.meshgrid( x_valsor,y_valsor)

        data_mx = self.data_df.to_numpy(dtype = np.complex64)


        cs = axis.contourf(x_matrix, y_matrix , data_mx,  cmap='jet', levels = levels )

        cs.changed()
        return cs
    
