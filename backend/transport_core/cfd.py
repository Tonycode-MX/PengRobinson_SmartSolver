import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict

# =====================================================================
# 1. SPATIAL MATHEMATICAL OPERATORS
# =====================================================================

def central_difference_x(f: np.ndarray, element_length: float) -> np.ndarray:
    """
    Computes the central difference with respect to 'x' of a 2D scalar or vector field.
    
    Args:
        f (np.ndarray): 2D array representing the physical field.
        element_length (float): Grid element size (dx).
        
    Returns:
        np.ndarray: 2D array with the spatial derivative in x.
    """
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, 0:-2]) / (2 * element_length)
    return diff

def central_difference_y(f: np.ndarray, element_length: float) -> np.ndarray:
    """
    Computes the central difference with respect to 'y' of a 2D scalar or vector field.
    
    Args:
        f (np.ndarray): 2D array representing the physical field.
        element_length (float): Grid element size (dy).
        
    Returns:
        np.ndarray: 2D array with the spatial derivative in y.
    """
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (f[2:, 1:-1] - f[0:-2, 1:-1]) / (2 * element_length)
    return diff

def laplacian(f: np.ndarray, element_length: float) -> np.ndarray:
    """
    Computes the 2D Laplacian operator using central finite differences.
    
    Args:
        f (np.ndarray): 2D array representing the physical field.
        element_length (float): Grid element size (dx = dy).
        
    Returns:
        np.ndarray: 2D array with the Laplacian field applied to f.
    """
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (
        f[1:-1, 2:] + f[1:-1, 0:-2] + f[2:, 1:-1] + f[0:-2, 1:-1] - 4 * f[1:-1, 1:-1]
    ) / (element_length ** 2)
    return diff


# =====================================================================
# 2. FLUID SIMULATION ENGINE (API Ready)
# =====================================================================

def solve_navier_stokes(
    n_points: int = 41,
    domain_size: float = 1.0,
    n_iterations: int = 500,
    time_step_length: float = 0.0001,
    kinematic_viscosity: float = 0.01,
    density: float = 1.0,
    horizontal_velocity: float = 1.0,
    n_pressure_poisson_iterations: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves the 2D incompressible Navier-Stokes equations for a lid-driven cavity flow.
    
    Args:
        n_points (int): Number of grid points in each dimension.
        domain_size (float): Physical length of the square domain.
        n_iterations (int): Number of time steps to simulate.
        time_step_length (float): Time step size (dt).
        kinematic_viscosity (float): Kinematic viscosity of the fluid.
        density (float): Density of the fluid.
        horizontal_velocity (float): Prescribed velocity at the top wall.
        n_pressure_poisson_iterations (int): Iterations for the pressure Poisson solver.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        (X, Y, u_next, v_next, p_next) representing the mesh coordinates and final fields.
        
    Raises:
        ValueError: If the input parameters are physically invalid.
    """
    if n_points <= 2:
        raise ValueError("n_points must be greater than 2.")
    if time_step_length <= 0:
        raise ValueError("time_step_length must be positive.")

    element_length = domain_size / (n_points - 1)
    x = np.linspace(0.0, domain_size, n_points)
    y = np.linspace(0.0, domain_size, n_points)
    X, Y = np.meshgrid(x, y)
    
    u_prev = np.zeros_like(X)
    v_prev = np.zeros_like(X)
    p_prev = np.zeros_like(X)

    for _ in tqdm(range(n_iterations), desc="Computing Fluid Flow"):
        du_prev_dx = central_difference_x(u_prev, element_length)
        du_prev_dy = central_difference_y(u_prev, element_length)
        dv_prev_dx = central_difference_x(v_prev, element_length)
        dv_prev_dy = central_difference_y(v_prev, element_length)
        
        laplace_u_prev = laplacian(u_prev, element_length)
        laplace_v_prev = laplacian(v_prev, element_length)
        
        # 2. Tentative step (solving the momentum equation without the pressure gradient term)
        u_tentative = (
            u_prev + 
            time_step_length * (-u_prev * du_prev_dx - v_prev * du_prev_dy + kinematic_viscosity * laplace_u_prev)
        )
        
        v_tentative = (
            v_prev + 
            time_step_length * (-u_prev * dv_prev_dx - v_prev * dv_prev_dy + kinematic_viscosity * laplace_v_prev)
        )

        u_tentative[0, :] = 0.0
        u_tentative[:, 0] = 0.0
        u_tentative[:, -1] = 0.0
        u_tentative[-1, :] = horizontal_velocity
        
        v_tentative[0, :] = 0.0
        v_tentative[:, 0] = 0.0
        v_tentative[:, -1] = 0.0
        v_tentative[-1, :] = 0.0

        du_tent_dx = central_difference_x(u_tentative, element_length)
        dv_tent_dy = central_difference_y(v_tentative, element_length)
        
        rhs = (density / time_step_length * (du_tent_dx + dv_tent_dy))

        for _ in range(n_pressure_poisson_iterations):
            p_next = np.zeros_like(p_prev)
            p_next[1:-1, 1:-1] = 1/4 * (p_prev[1:-1, 0:-2] + p_prev[0:-2, 1:-1] + p_prev[1:-1, 2:] + p_prev[2:, 1:-1] - element_length**2 * rhs[1:-1, 1:-1])
            
            p_next[:, -1] = p_next[:, -2]
            p_next[0, :] = p_next[1, :]
            p_next[:, 0] = p_next[:, 1] 
            p_next[-1, :] = 0.0
            p_prev = p_next

        dp_next_dx = central_difference_x(p_next, element_length)
        dp_next_dy = central_difference_y(p_next, element_length)

        u_next = (u_tentative - time_step_length / density * dp_next_dx) 
        v_next = (v_tentative - time_step_length / density * dp_next_dy)

        u_next[0, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0
        u_next[-1, :] = horizontal_velocity
        
        v_next[0, :] = 0.0
        v_next[:, 0] = 0.0
        v_next[:, -1] = 0.0
        v_next[-1, :] = 0.0
        
        u_prev = u_next
        v_prev = v_next
        p_prev = p_next

    return X, Y, u_next, v_next, p_next


# =====================================================================
# 3. DATA EXTRACTION FOR API / TABLE
# =====================================================================

def get_centerline_data(X: np.ndarray, u: np.ndarray, v: np.ndarray, p: np.ndarray) -> List[Dict[str, float]]:
    """
    Extracts the horizontal centerline data (middle of the domain) to be returned 
    as a JSON-like list of dictionaries for APIs, or to be printed as a table.
    
    Args:
        X (np.ndarray): Mesh X coordinates.
        u (np.ndarray): Horizontal velocity field.
        v (np.ndarray): Vertical velocity field.
        p (np.ndarray): Pressure field.
        
    Returns:
        List[Dict[str, float]]: A list of dictionaries containing numerical results.
    """
    mid_idx = u.shape[0] // 2  # Find the middle row
    
    results = []
    for i in range(u.shape[1]):
        row_data = {
            "x_coord": round(float(X[mid_idx, i]), 4),
            "u_velocity": round(float(u[mid_idx, i]), 4),
            "v_velocity": round(float(v[mid_idx, i]), 4),
            "pressure": round(float(p[mid_idx, i]), 4)
        }
        results.append(row_data)
        
    return results

def print_table(data: List[Dict[str, float]]) -> None:
    """
    Prints the extracted API data as a formatted console table.
    """
    print(f"\n{'X Coord':<10} | {'U Vel':<10} | {'V Vel':<10} | {'Pressure':<10}")
    print("-" * 50)
    for row in data:
        print(f"{row['x_coord']:<10.4f} | {row['u_velocity']:<10.4f} | {row['v_velocity']:<10.4f} | {row['pressure']:<10.4f}")
    print("-" * 50)


# =====================================================================
# 4. VISUALIZATION MODULE
# =====================================================================

def plot_flow_results(X: np.ndarray, Y: np.ndarray, u: np.ndarray, v: np.ndarray, p: np.ndarray) -> None:
    """
    Plots the fluid simulation results using matplotlib.
    """
    plt.figure(figsize=(8, 6)) 
    contour = plt.contourf(X, Y, p, cmap="viridis", alpha=0.8)
    plt.colorbar(contour, label="Pressure")
    plt.quiver(X, Y, u, v, color="black")
    
    plt.title("Lid-Driven Cavity Flow Simulation")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()  # This MUST trigger the window to open


# =====================================================================
# MAIN EXECUTION 
# =====================================================================

if __name__ == "__main__":
    # 1. SOLVE: Compute the simulation (Takes a few seconds, look at the progress bar)
    print("Starting simulation...")
    X, Y, u_final, v_final, p_final = solve_navier_stokes()
    
    # 2. TABULATE: Get numeric data for the API and print it
    print("\nExtracting numerical results (Horizontal Centerline):")
    api_data_response = get_centerline_data(X, u_final, v_final, p_final)
    print_table(api_data_response)
    
    # 3. PLOT: Render the visual graphics
    print("\nRendering plots. Close the plot window to end the script...")
    plot_flow_results(X, Y, u_final, v_final, p_final)


