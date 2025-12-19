"""
Plotter - Comprehensive plotting class for Jahn-Teller dynamics visualizations.

This class consolidates all plotting functionality for:
- Energy states vs magnetic field
- APES (Adiabatic Potential Energy Surface) plots
- ZPL (Zero Phonon Line) calculations
- LzSz expectation values
- Transition energies
- Contour plots
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Any
from matplotlib.colors import LogNorm
import os

import jahn_teller_dynamics.physics.jahn_teller_theory as jt
from jahn_teller_dynamics.physics.jahn_teller_theory import Jahn_Teller_Theory

# Constant for magnetic field column name
MAG_FIELD_STRENGTH_CSV_COL = 'magnetic field (T)'


class Plotter:
    """
    Comprehensive plotting class for Jahn-Teller dynamics visualizations.
    
    This class provides methods for all types of plots used in Jahn-Teller
    calculations, with optional PathManager integration for automatic file saving.
    """
    
    def __init__(
        self,
        path_manager: Optional[Any] = None,
        default_font_size: int = 14,
        default_dpi: int = 300,
        show_plots: bool = True
    ):
        """
        Initialize the plotter.
        
        Args:
            path_manager: Optional PathManager instance for automatic path handling
            default_font_size: Default font size for plots
            default_dpi: Default DPI for saved figures
            show_plots: Whether to show plots automatically (default: True)
        """
        self.path_manager = path_manager
        self.default_font_size = default_font_size
        self.default_dpi = default_dpi
        self.show_plots = show_plots
        plt.rcParams['font.size'] = default_font_size
    
    # ==================== Energy States Plots ====================
    
    def plot_energy_states_vs_magnetic_field(
        self,
        Es_dict: Dict[str, List[float]],
        field_strengths: Optional[List[float]] = None,
        title: str = 'Energy States vs Magnetic Field',
        xlabel: str = 'Magnetic Field (T)',
        ylabel: str = 'Energy (GHz)',
        save_path: Optional[str] = None,
        figsize: tuple = (8, 6)
    ) -> plt.Figure:
        """
        Plot energy states as a function of magnetic field.
        
        Args:
            Es_dict: Dictionary with energy state labels and energy lists
            field_strengths: List of field strengths (if None, uses Es_dict key)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Optional path to save figure
            figsize: Figure size tuple
            
        Returns:
            plt.Figure: The figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get field strengths
        if field_strengths is None:
            if MAG_FIELD_STRENGTH_CSV_COL in Es_dict:
                field_strengths = Es_dict[MAG_FIELD_STRENGTH_CSV_COL]
            else:
                raise ValueError("field_strengths must be provided or in Es_dict")
        
        # Plot each energy state
        energy_keys = [k for k in Es_dict.keys() if k != MAG_FIELD_STRENGTH_CSV_COL]
        for key in energy_keys[:4]:  # Limit to first 4 states
            values = Es_dict[key]
            ax.plot(field_strengths, values, label=key)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        
        if save_path:
            self._save_figure(fig, save_path)
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    # ==================== APES Plots ====================
    
    def plot_3D_APES(
        self,
        jt_theory: Jahn_Teller_Theory,
        r_range: tuple = (-1.0, 1.0),
        phi_range: tuple = (0, 2*np.pi),
        n_points: int = 1000,
        cmap: str = 'YlGnBu_r',
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8)
    ) -> plt.Figure:
        """
        Plot 3D Adiabatic Potential Energy Surface.
        
        Args:
            jt_theory: Jahn_Teller_Theory object
            r_range: Range for radial coordinate
            phi_range: Range for angular coordinate
            n_points: Number of points for meshgrid
            cmap: Colormap name
            save_path: Optional path to save figure
            figsize: Figure size tuple
            
        Returns:
            plt.Figure: The figure object
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        
        F = jt_theory.F
        G = jt_theory.G
        K = jt_theory.K
        
        r = np.linspace(r_range[0], r_range[1], n_points)
        phi = np.linspace(phi_range[0], phi_range[1], n_points)
        
        R, Phi = np.meshgrid(r, phi)
        
        Z1 = 0.5*K*R**2 + R*(F**2 + G**2*R**2 + 2*F*G*R*np.cos(3*Phi))**0.5
        Z2 = 0.5*K*R**2 - R*(F**2 + G**2*R**2 + 2*F*G*R*np.cos(3*Phi))**0.5
        
        X, Y = R*np.cos(Phi), R*np.sin(Phi)
        
        ax.plot_surface(X, Y, Z1, cmap=plt.cm.get_cmap(cmap), alpha=0.8)
        ax.plot_surface(X, Y, Z2, cmap=plt.cm.get_cmap(cmap), alpha=0.8)
        
        ax.set_xlabel(r'$\phi_\mathrm{real}$')
        ax.set_ylabel(r'$\phi_\mathrm{im}$')
        ax.set_zlabel(r'$V(\phi)$')
        ax.set_title('3D Adiabatic Potential Energy Surface')
        
        if save_path:
            self._save_figure(fig, save_path)
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_2D_APES(
        self,
        jt_theory: Jahn_Teller_Theory,
        x_range: tuple = (-1, 1),
        n_points: int = 1000,
        show_theoretical: bool = True,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8)
    ) -> plt.Figure:
        """
        Plot 2D Adiabatic Potential Energy Surface.
        
        Args:
            jt_theory: Jahn_Teller_Theory object
            x_range: Range for x coordinate
            n_points: Number of points
            show_theoretical: Whether to show theoretical predictions
            save_path: Optional path to save figure
            figsize: Figure size tuple
            
        Returns:
            plt.Figure: The figure object
        """
        plt.rcParams['font.size'] = self.default_font_size
        
        xs = np.linspace(x_range[0], x_range[1], n_points)
        
        K = jt_theory.K
        F, G = jt_theory.F, jt_theory.G
        
        ys2 = 0.5*K*xs**2 + xs*(F**2 + G**2*xs**2 + 2*F*G*xs*np.cos(3*0))**0.5
        ys1 = 0.5*K*xs**2 - xs*(F**2 + G**2*xs**2 + 2*F*G*xs*np.cos(3*0))**0.5
        
        jt_dist = jt_theory.JT_dist
        barr_dist = jt_theory.barrier_dist
        E_JT = -jt_theory.E_JT_meV
        E_barr_en_latt = E_JT - jt_theory.delta_meV
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(xs, ys1, label='Lower branch')
        ax.plot(xs, ys2, label='Upper branch')
        
        ax.plot([-jt_dist], [E_JT], 'x', markersize=10, label='Jahn-Teller energy')
        ax.plot([barr_dist], [E_barr_en_latt], 'x', markersize=10, 
                label="Jahn-Teller energy - Barrier energy")
        
        if show_theoretical:
            jt_dist_th = -F/(K+2*G)
            barr_dist_th = F/(K-2*G)
            ax.plot([jt_dist_th], [E_JT], 'x', markersize=10,
                    label='Jahn-Teller energy (Bersuker 3.28 eq.)')
            ax.plot([barr_dist_th], [E_barr_en_latt], 'x', markersize=10,
                    label="Jahn-Teller energy - Barrier energy (Bersuker 3.28 eq.)")
        
        ax.set_xlabel('distance (normal coordinates)')
        ax.set_ylabel('energy (meV)')
        ax.set_title('2D Adiabatic Potential Energy Surface')
        ax.legend()
        ax.grid(alpha=0.3)
        
        if save_path:
            self._save_figure(fig, save_path)
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_APES_simple(
        self,
        F: float,
        G: float,
        x_range: tuple = (-100, 100),
        y_range: tuple = (-100, 100),
        n_points: int = 100,
        cmap: str = 'cool',
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8)
    ) -> plt.Figure:
        """
        Plot simple APES from F and G coefficients.
        
        Args:
            F: F coefficient
            G: G coefficient
            x_range: Range for x coordinate
            y_range: Range for y coordinate
            n_points: Number of points for meshgrid
            cmap: Colormap name
            save_path: Optional path to save figure
            figsize: Figure size tuple
            
        Returns:
            plt.Figure: The figure object
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        
        def f(x, y):
            return F*(x+y) + G*((x**2-y**2) - 2*x*y)
        
        Z = f(X, Y)
        
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection='3d')
        
        ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8)
        
        ax.set_title('APES', fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
        
        if save_path:
            self._save_figure(fig, save_path)
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    # ==================== LzSz Plots ====================
    
    def plot_LzSz_expectation_values(
        self,
        LzSz_expected_vals: List[float],
        eigen_energies: List[float],
        title: str = 'Spin-orbit coupling \n expectation value',
        xlabel: str = r'$\left< L_{z} \otimes S_{z} \right>$',
        ylabel: str = r'Energy (meV)',
        save_path: Optional[str] = None,
        figsize: tuple = (8, 6)
    ) -> plt.Figure:
        """
        Plot LzSz expectation values vs eigenenergies.
        
        Args:
            LzSz_expected_vals: List of LzSz expectation values
            eigen_energies: List of eigenenergies
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Optional path to save figure
            figsize: Figure size tuple
            
        Returns:
            plt.Figure: The figure object
        """
        plt.rcParams['font.size'] = 18
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(LzSz_expected_vals, eigen_energies, 'x', markersize=8)
        ax.plot([0.0, 0.0], [eigen_energies[0], eigen_energies[-1]], '--')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid()
        
        if save_path:
            self._save_figure(fig, save_path)
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    # ==================== ZPL Plots ====================
    
    def plot_ZPL_transitions(
        self,
        A_transition: Dict[str, List[float]],
        B_transition: Dict[str, List[float]],
        C_transition: Dict[str, List[float]],
        D_transition: Dict[str, List[float]],
        field_strengths: List[float],
        calculation_name: str = '',
        save_path: Optional[str] = None,
        figsize: tuple = (14, 10),
        dpi: int = 700,
        B_min: Optional[float] = None,
        B_max: Optional[float] = None
    ) -> plt.Figure:
        """
        Plot ZPL transitions in 4-panel layout.
        
        Args:
            A_transition: Dictionary with A transition data
            B_transition: Dictionary with B transition data
            C_transition: Dictionary with C transition data
            D_transition: Dictionary with D transition data
            field_strengths: List of magnetic field strengths
            calculation_name: Name for the calculation
            save_path: Optional path to save figure
            figsize: Figure size tuple
            dpi: DPI for saved figure
            B_min: Optional minimum magnetic field for x-axis limits
            B_max: Optional maximum magnetic field for x-axis limits
            
        Returns:
            plt.Figure: The figure object
        """
        plt.rcParams['font.size'] = 20
        
        line_labels = ['line_0 (GHz)', 'line_1 (GHz)', 'line_2 (GHz)', 'line_3 (GHz)']
        
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=figsize)
        
        title_name = calculation_name.replace('_', ' ')
        fig.suptitle(title_name)
        
        # Calculate energy shift for zero line
        energy_shift = (A_transition[line_labels[0]][0] - D_transition[line_labels[0]][0]) / 2
        zeroline = abs(A_transition[line_labels[0]][0] - energy_shift)
        
        # Set x-axis limits
        if B_min is not None and B_max is not None:
            [ax.set_xlim(B_min, B_max) for ax in axes]
        else:
            B_min = min(field_strengths)
            B_max = max(field_strengths)
            [ax.set_xlim(B_min, B_max) for ax in axes]
        
        [ax.tick_params(labeltop=False, bottom=False, labelright=True, right=True) for ax in axes]
        
        axes[3].xaxis.tick_bottom()
        axes[2].annotate('ZPL shift (GHz)', (-0.12, 0.45), xycoords='axes fraction', rotation=90)
        axes[3].set_xlabel('magnetic field (T)')
        
        # Plot transitions
        for line_label in line_labels:
            axes[0].plot(field_strengths, np.array(A_transition[line_label]) + zeroline, '-k')
            axes[1].plot(field_strengths, np.array(B_transition[line_label]) + zeroline, '-k')
            axes[2].plot(field_strengths, np.array(C_transition[line_label]) + zeroline, '-k')
            axes[3].plot(field_strengths, np.array(D_transition[line_label]) + zeroline, '-k')
        
        if save_path:
            self._save_figure(fig, save_path, dpi=dpi)
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_ZPL_transitions_to_output(
        self,
        A_transition: Dict[str, List[float]],
        B_transition: Dict[str, List[float]],
        C_transition: Dict[str, List[float]],
        D_transition: Dict[str, List[float]],
        field_strengths: List[float],
        suffix: str = '_ZPL_calculation.png',
        dpi: int = 700
    ) -> str:
        """
        Plot ZPL transitions and save to output folder with automatic naming.
        
        Args:
            A_transition: Dictionary with A transition data
            B_transition: Dictionary with B transition data
            C_transition: Dictionary with C transition data
            D_transition: Dictionary with D transition data
            field_strengths: List of magnetic field strengths
            suffix: Filename suffix
            dpi: DPI for saved figure
            
        Returns:
            str: Full filepath used
        """
        calculation_name = self._get_prefix_name() if self.path_manager else ''
        filename = self._get_prefixed_filename(suffix) if self.path_manager else suffix
        filepath = self._get_output_path(filename) if self.path_manager else filename
        
        self.ensure_directory(filepath)
        
        self.plot_ZPL_transitions(
            A_transition, B_transition, C_transition, D_transition,
            field_strengths, calculation_name, save_path=filepath, dpi=dpi
        )
        
        return filepath
    
    # ==================== ZPL Intensity Plots ====================
    
    def plot_ZPL_intensity_scatter(
        self,
        magnetic_field: np.ndarray,
        energy: np.ndarray,
        intensity: np.ndarray,
        title: str = 'ZPL Intensity vs. Magnetic Field and Energy',
        xlabel: str = 'Magnetic Field (T)',
        ylabel: str = 'Energy (eV)',
        cmap: str = 'jet',
        save_path: Optional[str] = None,
        figsize: tuple = (8, 6)
    ) -> plt.Figure:
        """
        Plot ZPL intensity as a function of magnetic field and energy.
        
        Args:
            magnetic_field: Magnetic field values in Tesla
            energy: Energy values in eV
            intensity: 2D array of intensity values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            cmap: Colormap name
            save_path: Optional path to save figure
            figsize: Figure size tuple
            
        Returns:
            plt.Figure: The figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sc = ax.scatter(
            magnetic_field, energy, c=intensity,
            s=10,
            cmap=cmap,
            edgecolor='k',
            linewidth=0.0
        )
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        cbar = plt.colorbar(sc, ax=ax, label="Intensity (a.u.)")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    # ==================== Contour Plots ====================
    
    def plot_contour_from_dataframe(
        self,
        data_df: pd.DataFrame,
        xlabel: str = '',
        ylabel: str = '',
        title: str = '',
        cmap: str = 'jet',
        n_levels: int = 10000,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8)
    ) -> plt.Figure:
        """
        Plot contour from DataFrame (handles complex column names).
        
        Args:
            data_df: DataFrame with complex column names
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            cmap: Colormap name
            n_levels: Number of contour levels
            save_path: Optional path to save figure
            figsize: Figure size tuple
            
        Returns:
            plt.Figure: The figure object
        """
        y_vals = np.array(data_df.index.to_list())
        
        cols = data_df.columns.to_list()
        cols = list(map(complex, cols))
        x_vals = np.array(list(map(lambda x: float(x.real), cols)))
        
        x_matrix, y_matrix = np.meshgrid(x_vals, y_vals)
        
        data_mx = data_df.to_numpy(dtype=np.complex64)
        
        max_val = np.max(data_mx.flatten())
        min_val = np.min(data_mx.flatten())
        levels = np.linspace(min_val, max_val, n_levels)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.set_xlim(x_vals[0], x_vals[-1])
        ax.set_ylim(y_vals[0], y_vals[-1])
        
        cs = ax.contourf(x_matrix, y_matrix, data_mx, cmap=cmap, levels=levels)
        cs.changed()
        
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        
        plt.colorbar(cs, ax=ax)
        
        if save_path:
            self._save_figure(fig, save_path)
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_polar_contour(
        self,
        values: np.ndarray,
        zeniths: List[float],
        azimuths: List[float],
        title: str = 'Polar Contour Plot',
        colorbar_label: str = 'Pixel reflectance',
        n_levels: int = 30,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8)
    ) -> plt.Figure:
        """
        Plot polar contour plot.
        
        Args:
            values: Array of values to plot
            zeniths: List of zenith angles
            azimuths: List of azimuth angles
            title: Plot title
            colorbar_label: Colorbar label
            n_levels: Number of contour levels
            save_path: Optional path to save figure
            figsize: Figure size tuple
            
        Returns:
            plt.Figure: The figure object
        """
        theta = np.radians(azimuths)
        zeniths = np.array(zeniths)
        values = np.array(values)
        values = values.reshape(len(azimuths), len(zeniths))
        
        r, theta = np.meshgrid(zeniths, np.radians(azimuths))
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        
        cax = ax.contourf(theta, r, values, n_levels, cmap='jet')
        cax.changed()
        cb = fig.colorbar(cax, ax=ax)
        cb.set_label(colorbar_label)
        ax.set_title(title)
        
        if save_path:
            self._save_figure(fig, save_path)
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    # ==================== Utility Methods ====================
    
    def _get_output_path(self, filename: str) -> str:
        """Get full output path using PathManager if available."""
        if self.path_manager is None:
            return filename
        
        if os.path.dirname(filename):
            return filename
        
        output_folder = self.path_manager.get_res_folder_name()
        return output_folder + filename if output_folder else filename
    
    def _get_prefixed_filename(self, suffix: str) -> str:
        """Get filename with prefix from PathManager if available."""
        if self.path_manager is None:
            return suffix.lstrip('_') if suffix.startswith('_') else suffix
        
        prefix = self.path_manager.get_prefix_name()
        if prefix:
            return prefix + suffix
        return suffix.lstrip('_') if suffix.startswith('_') else suffix
    
    def _get_prefix_name(self) -> str:
        """Get prefix name from PathManager if available."""
        if self.path_manager:
            return self.path_manager.get_prefix_name()
        return ''
    
    def ensure_directory(self, filepath: str) -> None:
        """Ensure the directory for a filepath exists."""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
    
    def _save_figure(self, fig: plt.Figure, filepath: str, dpi: Optional[int] = None) -> None:
        """
        Save figure to filepath.
        
        Args:
            fig: Figure object to save
            filepath: Path to save figure
            dpi: DPI for saved figure (uses default if None)
        """
        self.ensure_directory(filepath)
        dpi = dpi if dpi is not None else self.default_dpi
        fig.savefig(filepath, bbox_inches='tight', dpi=dpi)
        print(f'Saving figure to {filepath}')
    
    def set_show_plots(self, show: bool) -> None:
        """Set whether to show plots automatically."""
        self.show_plots = show
    
    def set_font_size(self, size: int) -> None:
        """Set default font size."""
        self.default_font_size = size
        plt.rcParams['font.size'] = size
    
    def set_dpi(self, dpi: int) -> None:
        """Set default DPI for saved figures."""
        self.default_dpi = dpi

