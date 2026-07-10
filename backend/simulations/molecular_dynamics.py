import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.io as sio
import plotly.io as pio
from typing import Tuple

# =====================================================================
# 1. CORE PHYSICS FUNCTIONS (WITH DOCSTRINGS, TYPE HINTS & VALIDATION)
# =====================================================================
def calculate_energy(positions: np.ndarray, L: float) -> float:
    """
    Calculates the total potential energy of the system using a Lennard-Jones potential.
    
    The function applies the minimum image convention to simulate infinite periodic 
    boundary conditions and uses a hard-core cutoff to prevent unphysical particle overlap.

    Args:
        positions (np.ndarray): A 2D numpy array of shape (N, 3) containing the Cartesian 
                                coordinates of all particles in the system.
        L (float): Length of the cubic simulation box.

    Returns:
        float: The total potential energy of the current microstate.
        
    Raises:
        ValueError: If the simulation box length (L) is less than or equal to 0.
        TypeError: If the 'positions' argument is not a numpy array.
    """
    if not isinstance(positions, np.ndarray):
        raise TypeError("The 'positions' parameter must be a NumPy array.")
    if L <= 0:
        raise ValueError("Simulation box length (L) must be strictly positive.")

    n_particles = len(positions)
    energy = 0.0
    
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            rij = positions[i] - positions[j]
            rij -= L * np.round(rij / L) # Minimum image convention
            r = np.linalg.norm(rij)
            
            if r < 0.8: 
                energy += 1e6 # Hard core repulsion penalty
            elif r < 3.0: 
                sr6 = (1.0 / r)**6
                sr12 = sr6**2
                energy += 4 * (sr12 - sr6) # Lennard-Jones potential formula
                
    return energy

def calculate_properties(positions: np.ndarray, L: float, T: float) -> Tuple[float, float]:
    """
    Calculates the total Potential Energy and Real Pressure of the system using the Virial Theorem.

    Args:
        positions (np.ndarray): A 2D numpy array of shape (N, 3) containing particle coordinates.
        L (float): Length of the simulation box.
        T (float): System temperature (must be greater than 0).

    Returns:
        Tuple[float, float]: A tuple containing:
                             - Total potential energy (float).
                             - Real pressure of the system (float).
                             
    Raises:
        ValueError: If the box length (L) or temperature (T) are not strictly positive.
        TypeError: If the 'positions' argument is not a numpy array.
    """
    if not isinstance(positions, np.ndarray):
        raise TypeError("The 'positions' parameter must be a NumPy array.")
    if L <= 0:
        raise ValueError("Simulation box length (L) must be strictly positive.")
    if T <= 0:
        raise ValueError("Temperature (T) must be strictly positive.")

    n_particles = len(positions)
    energy = 0.0
    virial = 0.0
    volume = L**3
    
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            rij = positions[i] - positions[j]
            rij -= L * np.round(rij / L) # Minimum image convention
            r = np.linalg.norm(rij)
            
            if r < 0.8:
                energy += 1e6 
            elif r < 3.0: 
                sr6 = (1.0 / r)**6
                sr12 = sr6**2
                energy += 4 * (sr12 - sr6)
                virial += 24 * (2 * sr12 - sr6) # Virial calculation for pressure
                
    rho = n_particles / volume
    real_pressure = (rho * T) + (virial / (3.0 * volume))
    
    return energy, real_pressure

def calculate_gr(positions: np.ndarray, L: float, dr: float = 0.1, r_max: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Radial Distribution Function, g(r), to determine the structural organization of the particles.

    Args:
        positions (np.ndarray): Array of shape (N, 3) containing particle coordinates.
        L (float): Length of the simulation box.
        dr (float, optional): Bin width (shell thickness) for the histogram. Defaults to 0.1.
        r_max (float, optional): Maximum radius to evaluate. Defaults to 5.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
                                       - r_values: The center distance of each calculated bin.
                                       - gr_values: The normalized g(r) probability values.
                                       
    Raises:
        ValueError: If L, dr, or r_max are less than or equal to 0, or if r_max > L/2.
    """
    if L <= 0 or dr <= 0 or r_max <= 0:
        raise ValueError("L, dr, and r_max parameters must be strictly positive.")
    if r_max > L / 2.0:
        print("Warning: r_max is greater than half the box length (L/2). Minimum image convention artifacts may appear.")

    n_particles = len(positions)
    volume = L**3
    rho = n_particles / volume
    
    distances = []
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            rij = positions[i] - positions[j]
            rij -= L * np.round(rij / L)
            r = np.linalg.norm(rij)
            if r < r_max:
                distances.append(r)
                
    bins = np.arange(0, r_max + dr, dr)
    hist, bin_edges = np.histogram(distances, bins=bins)
    
    gr = []
    r_values = []
    
    for i in range(len(hist)):
        r_int = bin_edges[i]
        r_ext = bin_edges[i+1]
        r_mid = (r_int + r_ext) / 2.0
        r_values.append(r_mid)
        
        # Calculate the volume of the spherical shell
        shell_volume = (4.0 / 3.0) * np.pi * (r_ext**3 - r_int**3)
        ideal_n = 0.5 * n_particles * rho * shell_volume
        
        # Normalize the histogram counts by the ideal gas expectation
        if ideal_n > 0:
            gr.append(hist[i] / ideal_n)
        else:
            gr.append(0.0)
            
    return np.array(r_values), np.array(gr)

# =====================================================================
# 2. MAIN EXECUTION & WORKFLOW FUNCTION
# =====================================================================
def simulate(
    N: int = 64,
    L: float = 10.0,
    steps: int = 2000,
    delta: float = 0.2,
    T: float = 1.2
) -> None:
    """
    Executes a complete Monte Carlo simulation for a Lennard-Jones fluid.
    Includes an energy minimization phase, a thermodynamic property calculation phase,
    2D plotting via Matplotlib, data exporting, and a 3D visualization via Plotly.

    Args:
        N (int, optional): Number of particles. Defaults to 64.
        L (float, optional): Simulation box length. Defaults to 10.0.
        steps (int, optional): Number of Monte Carlo sweeps/steps. Defaults to 2000.
        delta (float, optional): Maximum displacement per coordinate. Defaults to 0.2.
        T (float, optional): System temperature. Defaults to 1.2.

    Returns:
        None
        
    Raises:
        ValueError: If physical or simulation parameters are unphysical (e.g., negative).
    """
    
    # --- Input Validation ---
    if N <= 1:
        raise ValueError("Error: Number of particles (N) must be greater than 1.")
    if steps <= 0:
        raise ValueError("Error: Simulation steps must be a strictly positive integer.")
    if L <= 0 or delta <= 0 or T <= 0:
        raise ValueError("Error: Physical parameters (L, delta, T) must be strictly positive.")

    # --- Initializing the Cubic Lattice ---
    n_per_side = int(np.ceil(N**(1/3)))
    x = np.linspace(0, L, n_per_side, endpoint=False)
    pos = np.array([[xi, xj, xk] for xi in x for xj in x for xk in x][:N])
    pos += np.random.uniform(-0.1, 0.1, size=pos.shape) # Add slight random noise
    
    energy_history = []

    # =================================================================
    # PHASE 1: ENERGY MINIMIZATION
    # =================================================================
    print("1/2 Executing Phase 1: Energy minimization simulation...")
    e_current = calculate_energy(pos, L)

    for step in range(steps):
        idx = np.random.randint(N)
        old_pos = pos[idx].copy()
        
        # Trial move
        pos[idx] += np.random.uniform(-delta, delta, 3)
        pos[idx] %= L
        
        e_new = calculate_energy(pos, L)
        
        # Metropolis acceptance criterion
        if e_new < e_current or np.random.rand() < np.exp(-(e_new - e_current)):
            e_current = e_new
        else:
            pos[idx] = old_pos # Reject and revert
            
        energy_history.append(e_current)

    # =================================================================
    # PHASE 2: THERMODYNAMIC EQUILIBRIUM & PRESSURE
    # =================================================================
    print("2/2 Executing Phase 2: Thermodynamic equilibrium and pressure calculation...")
    beta = 1.0 / T
    pressure_history = []
    pot_energy_history = []

    e_current, p_current = calculate_properties(pos, L, T)

    for step in range(steps):
        idx = np.random.randint(N)
        old_pos = pos[idx].copy()
        
        pos[idx] += np.random.uniform(-delta, delta, 3)
        pos[idx] %= L
        
        e_new, p_new = calculate_properties(pos, L, T)
        delta_e = e_new - e_current
        
        # Metropolis acceptance criterion for NVT ensemble
        if delta_e < 0 or np.random.rand() < np.exp(-beta * delta_e):
            e_current = e_new
            p_current = p_new
        else:
            pos[idx] = old_pos 
            
        pot_energy_history.append(e_current)
        pressure_history.append(p_current)

    # =================================================================
    # 3. MATPLOTLIB WINDOW (2D GRAPHS) - PREPARATION
    # =================================================================
    print("Preparing 2D graphs...")
    cutoff_step = 200 # Discard initial steps for equilibrium analysis

    plt.figure(figsize=(12, 8))

    # Plot 1: Total Energy Evolution
    plt.subplot(2, 2, 1)
    plt.plot(energy_history, color='blue', lw=1)
    plt.title("1. Total Energy Evolution")
    plt.xlabel("Monte Carlo Steps")
    plt.ylabel("Potential Energy")
    plt.grid(True)

    # Plot 2: Energy Fluctuations (Equilibrium phase)
    plt.subplot(2, 2, 2)
    y_data = energy_history[cutoff_step:]
    x_data = range(cutoff_step, cutoff_step + len(y_data))
    plt.plot(x_data, y_data, color='green', lw=1.5)
    plt.title("2. Energy Fluctuations (Equilibrium)")
    plt.xlabel("Monte Carlo Steps")
    plt.ylabel("Potential Energy")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Pressure Evolution
    plt.subplot(2, 2, 3)
    plt.plot(pressure_history, color='darkorange', lw=1, alpha=0.7)
    p_mean = np.mean(pressure_history[cutoff_step:])
    plt.axhline(p_mean, color='red', linestyle='--', lw=2, label=f'Mean P = {p_mean:.2f}')
    plt.title("3. Virial Pressure Evolution")
    plt.xlabel("Monte Carlo Steps")
    plt.ylabel("Pressure")
    plt.legend()
    plt.grid(True)

    # Plot 4: Radial Distribution Function g(r)
    plt.subplot(2, 2, 4)
    r_vals, gr_vals = calculate_gr(pos, L)
    plt.plot(r_vals, gr_vals, color='purple', lw=2)
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.7) # Ideal gas reference line
    plt.title("4. Radial Distribution Function g(r)")
    plt.xlabel("Distance (r)")
    plt.ylabel("g(r)")
    plt.xlim(0, 5.0)
    plt.grid(True)

    plt.tight_layout()

    # =================================================================
    # 4. FILE EXPORT & 3D BROWSER VISUALIZATION
    # =================================================================
    print("Exporting trajectory data for MATLAB...")
    mat_filename = 'gas_coordinates.mat'
    sio.savemat(mat_filename, {'positions': pos, 'energies': energy_history})
    print(f"SUCCESS! File '{mat_filename}' has been saved to your directory.")

    print("Launching 3D spatial visualization in the default web browser...")
    pio.renderers.default = "browser" 

    fig = go.Figure(data=[go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='markers',
        marker=dict(size=8, color=pos[:, 2], colorscale='Viridis', opacity=0.9),
        name="Particles"
    )])

    fig.update_layout(
        title="5. Final Molecular Configuration (3D)", 
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Trigger browser to open Plotly 3D graph (Does not pause code)
    fig.show()

    # Open Matplotlib 2D window and pause script
    print("Visualizations ready! Close the Matplotlib (2D) window to terminate the script.")
    plt.show() 

# =====================================================================
# 5. EXECUTION BLOCK
# =====================================================================
if __name__ == "__main__":
    simulate()