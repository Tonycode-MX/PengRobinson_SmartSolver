import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import scipy.io as sio
from typing import List

__all__ = ["calculate_energy", "plot_energy_graphs", "show_3d_simulation", "run_monte_carlo_simulation"]

# =====================================================================
# 1. ENERGY FUNCTION (LENNARD-JONES POTENTIAL)
# =====================================================================
def calculate_energy(positions: np.ndarray, box_length: float) -> float:
    """
    Calculates the total potential energy of the system using the Lennard-Jones potential.

    The Lennard-Jones potential between two particles is given by:
    $V(r) = 4\epsilon \\left[ \\left(\\frac{\\sigma}{r}\\right)^{12} - \\left(\\frac{\\sigma}{r}\\right)^6 \\right]$
    This function implements a simplified version with reduced units ($\epsilon = 1$, $\sigma = 1$).
    It applies the Minimum Image Convention for periodic boundary conditions and includes
    a harsh penalty for particle overlaps (simulating hard-sphere repulsion at very short distances).

    Args:
        positions (np.ndarray): A 2D array of shape (N, 3) containing the [x, y, z] 
                                coordinates of all particles in the system.
        box_length (float): The length of the cubic simulation box (L).

    Returns:
        float: The total calculated potential energy of the system.

    Raises:
        ValueError: If `box_length` is less than or equal to zero.
        ValueError: If the `positions` array is empty.
        ValueError: If the `positions` array does not have exactly 3 columns (3D space).
    """
    if box_length <= 0:
        raise ValueError("The 'box_length' must be greater than zero.")
    if positions.size == 0:
        raise ValueError("The 'positions' array cannot be empty.")
    if len(positions.shape) != 2 or positions.shape[1] != 3:
        raise ValueError("The 'positions' array must be a 2D array with exactly 3 columns for 3D coordinates.")

    n_particles: int = len(positions)
    total_energy: float = 0.0

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            rij: np.ndarray = positions[i] - positions[j]
            rij -= box_length * np.round(rij / box_length) 
            r: float = float(np.linalg.norm(rij))
            
            if r < 0.8:
                total_energy += 1e6  # Penalty if particles collide
            elif r < 3.0:            # Cutoff radius
                sr6: float = (1.0 / r)**6
                sr12: float = sr6**2
                total_energy += 4.0 * (sr12 - sr6)
                
    return total_energy

# =====================================================================
# 2. MONTE CARLO SIMULATION RUNNER
# =====================================================================

def run_monte_carlo_simulation(n_particles: int, box_length: float, steps: int, delta: float) -> dict:
    """
    AI TOOL: Executes a complete Monte Carlo thermodynamic simulation using the Lennard-Jones potential.
    
    Args:
        n_particles (int): Number of particles in the system (e.g., 64).
        box_length (float): Length of the cubic simulation box (e.g., 10.0).
        steps (int): Number of Monte Carlo steps to iterate (e.g., 2000).
        delta (float): Maximum displacement per step (e.g., 0.2).
        
    Returns:
        dict: A dictionary containing the final 'positions' (np.ndarray) and the 'energy_history' (List[float]).
    """
    energy_history: List[float] = []
    
    # Create an initial cubic lattice
    n_per_side: int = int(np.ceil(n_particles**(1/3)))
    x_coords: np.ndarray = np.linspace(0, box_length, n_per_side, endpoint=False)
    positions: np.ndarray = np.array([[xi, xj, xk] for xi in x_coords for xj in x_coords for xk in x_coords][:n_particles])
    positions += np.random.uniform(-0.1, 0.1, size=positions.shape)

    current_energy: float = calculate_energy(positions, box_length)

    for step in range(steps):
        idx: int = np.random.randint(n_particles)
        old_pos: np.ndarray = positions[idx].copy()
        
        positions[idx] += np.random.uniform(-delta, delta, 3)
        positions[idx] %= box_length
        
        new_energy: float = calculate_energy(positions, box_length)
        
        if new_energy < current_energy or np.random.rand() < np.exp(-(new_energy - current_energy)):
            current_energy = new_energy
        else:
            positions[idx] = old_pos
            
        energy_history.append(current_energy)
        
    return {
        "positions": positions,
        "energy_history": energy_history
    }

# =====================================================================
# 3. 2D MATPLOTLIB GRAPHING TOOL
# =====================================================================
def plot_energy_graphs(energy_data: List[float], cutoff: int) -> None:
    """
    AI TOOL: Use this function to plot the 2D energy evolution of the Monte Carlo simulation.
    
    This function creates a single Matplotlib window with two side-by-side graphs: 
    one showing the complete energy minimization over time, and a second zoomed-in 
    graph showing the thermodynamic equilibrium fluctuations after a specified cutoff step.

    Args:
        energy_data (List[float]): The complete list of energy values recorded during the simulation.
        cutoff (int): The step number from which to start the zoomed-in equilibrium plot 
                      (ignoring the initial energy drop).

    Returns:
        None: Displays the plot using matplotlib.pyplot.show() (handled outside the function).

    Raises:
        ValueError: If the `energy_data` list is empty.
        ValueError: If the `cutoff` is negative or greater than/equal to the length of `energy_data`.
    """
    if not energy_data:
        raise ValueError("The 'energy_data' list cannot be empty.")
    if cutoff < 0 or cutoff >= len(energy_data):
        raise ValueError("The 'cutoff' must be a positive integer less than the total number of steps.")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Graph 1: Complete Minimization
    axes[0].plot(energy_data, color='blue', lw=1)
    axes[0].set_title("1. Energy Evolution (Complete Minimization)")
    axes[0].set_xlabel("Monte Carlo Steps")
    axes[0].set_ylabel("Potential Energy")
    axes[0].grid(True)

    # Graph 2: Zoom on Fluctuations
    y_data: List[float] = energy_data[cutoff:]
    x_data: range = range(cutoff, cutoff + len(y_data))
    
    axes[1].plot(list(x_data), y_data, color='green', lw=1.5)
    axes[1].set_title("2. Fluctuations in Thermodynamic Equilibrium")
    axes[1].set_xlabel("Total Monte Carlo Steps")
    axes[1].set_ylabel("Potential Energy")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

# =====================================================================
# 4. 3D PLOTLY VISUALIZATION TOOL
# =====================================================================
def show_3d_simulation(pos_data: np.ndarray) -> None:
    """
    AI TOOL: Use this function to render an interactive 3D visualization of the molecules.
    
    Generates a 3D scatter plot of the final molecular configuration using Plotly. 
    It opens automatically in the user's default web browser.

    Args:
        pos_data (np.ndarray): A 2D array of shape (N, 3) containing the final 
                               [x, y, z] coordinates of the particles.

    Returns:
        None: Opens an interactive HTML plot in the default browser.

    Raises:
        ValueError: If the `pos_data` array does not have exactly 3 columns (3D coordinates).
        ValueError: If the `pos_data` array is empty.
    """
    if pos_data.size == 0:
        raise ValueError("The 'pos_data' array cannot be empty.")
    if len(pos_data.shape) != 2 or pos_data.shape[1] != 3:
        raise ValueError("The 'pos_data' array must be a 2D array with exactly 3 columns.")

    pio.renderers.default = "browser"
    
    fig = go.Figure(data=[go.Scatter3d(
        x=pos_data[:, 0], y=pos_data[:, 1], z=pos_data[:, 2],
        mode='markers',
        marker=dict(size=8, color=pos_data[:, 2], colorscale='Viridis', opacity=0.9)
    )])

    fig.update_layout(
        title="3. Final Molecular Configuration",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

# =====================================================================
# 4. MAIN EXECUTION BLOCK (PROTECTED FOR AI AGENT IMPORTS)
# =====================================================================
# The 'if __name__ == "__main__":' check ensures that if an AI agent imports 
# this file to read the tools, it won't accidentally trigger the 2000-step simulation.
if __name__ == "__main__":
    
    # --- Configuration ---
    N: int = 64
    L: float = 10.0
    steps: int = 2000
    delta: float = 0.2
    energy_history: List[float] = []

    # Create an initial cubic lattice
    n_per_side: int = int(np.ceil(N**(1/3)))
    x_coords: np.ndarray = np.linspace(0, L, n_per_side, endpoint=False)
    positions: np.ndarray = np.array([[xi, xj, xk] for xi in x_coords for xj in x_coords for xk in x_coords][:N])
    positions += np.random.uniform(-0.1, 0.1, size=positions.shape)

    # --- Simulation Loop ---
    current_energy: float = calculate_energy(positions, L)

    for step in range(steps):
        idx: int = np.random.randint(N)
        old_pos: np.ndarray = positions[idx].copy()
        
        positions[idx] += np.random.uniform(-delta, delta, 3)
        positions[idx] %= L
        
        new_energy: float = calculate_energy(positions, L)
        
        if new_energy < current_energy or np.random.rand() < np.exp(-(new_energy - current_energy)):
            current_energy = new_energy
        else:
            positions[idx] = old_pos
            
        energy_history.append(current_energy)

    # --- Simultaneous Execution of Graphing Tools ---
    cutoff_step: int = 200
    
    # 1. Trigger the 3D Plotly visualization (opens in browser, non-blocking)
    show_3d_simulation(positions)
    
    # 2. Trigger the 2D Matplotlib plots (blocks Python until user closes window)
    plot_energy_graphs(energy_history, cutoff_step)
    plt.show()

    # --- Export to MATLAB ---
    try:
        sio.savemat('gas_coordinates.mat', {'positions': positions, 'energies': energy_history})
        print("SUCCESS! File 'gas_coordinates.mat' saved and ready for MATLAB.")
    except Exception as e:
        raise ValueError(f"Failed to export data to MATLAB format. Error: {e}")