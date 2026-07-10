import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Tuple

# =====================================================================
# SPATIAL MATHEMATICAL OPERATORS
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
# FLUID SIMULATION ENGINE
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
        A tuple containing (X, Y, u_next, v_next, p_next) representing the mesh 
        coordinates and the final velocity and pressure fields.

    Raises:
        ValueError: If the input parameters are physically invalid.
    """
    # API Input Validations
    if n_points <= 2:
        raise ValueError("The number of grid points (n_points) must be greater than 2.")
    if domain_size <= 0:
        raise ValueError("The domain size (domain_size) must be positive.")
    if time_step_length <= 0:
        raise ValueError("The time step length (time_step_length) must be positive.")
    if kinematic_viscosity <= 0 or density <= 0:
        raise ValueError("Viscosity and density must be positive values.")
    if n_iterations <= 0 or n_pressure_poisson_iterations <= 0:
        raise ValueError("The number of iterations must be a positive integer.")

    # Mesh Configuration
    element_length = domain_size / (n_points - 1)
    x = np.linspace(0.0, domain_size, n_points)
    y = np.linspace(0.0, domain_size, n_points)
    X, Y = np.meshgrid(x, y)
    
    # Field Initialization
    u_prev = np.zeros_like(X)
    v_prev = np.zeros_like(X)
    p_prev = np.zeros_like(X)

    # Main time-stepping loop
    for _ in tqdm(range(n_iterations), desc="Solving Navier-Stokes"):
        
        # 1. Derivatives from the previous step
        du_prev_dx = central_difference_x(u_prev, element_length)
        du_prev_dy = central_difference_y(u_prev, element_length)
        dv_prev_dx = central_difference_x(v_prev, element_length)
        dv_prev_dy = central_difference_y(v_prev, element_length)
        
        laplace_u_prev = laplacian(u_prev, element_length)
        laplace_v_prev = laplacian(v_prev, element_length)
        
        # 2. Tentative step (solving the momentum equation without the pressure gradient term)
        u_tentative = (u_prev + time_step_length * (-u_prev * du_prev_dx + v_prev * du_prev_dy) + kinematic_viscosity * laplace_u_prev)
        v_tentative = (v_prev + time_step_length * (-u_prev * dv_prev_dx + v_prev * dv_prev_dy) + kinematic_viscosity * laplace_v_prev)

        # 3. Boundary conditions for the tentative velocity
        # Homogeneous Dirichlet everywhere except for the horizontal velocity at the top
        u_tentative[0, :] = 0.0
        u_tentative[:, 0] = 0.0
        u_tentative[:, -1] = 0.0
        u_tentative[-1, :] = horizontal_velocity
        
        v_tentative[0, :] = 0.0
        v_tentative[:, 0] = 0.0
        v_tentative[:, -1] = 0.0
        v_tentative[-1, :] = 0.0

        # 4. Poisson equation for pressure
        du_tent_dx = central_difference_x(u_tentative, element_length)
        dv_tent_dy = central_difference_y(v_tentative, element_length)
        
        rhs = (density / time_step_length * (du_tent_dx + dv_tent_dy))

        for _ in range(n_pressure_poisson_iterations):
            p_next = np.zeros_like(p_prev)
            p_next[1:-1, 1:-1] = 1/4 * (p_prev[1:-1, 0:-2] + p_prev[0:-2, 1:-1] + p_prev[1:-1, 2:] + p_prev[2:, 1:-1] - element_length**2 * rhs[1:-1, 1:-1])
            
            # Pressure Boundary Conditions: Homogeneous Neumann everywhere except the top (Dirichlet)
            p_next[:, -1] = p_next[:, -2]
            p_next[0, :] = p_next[1, :]
            p_next[:, 0] = p_next[:, 1] 
            p_next[-1, :] = 0.0

            p_prev = p_next

        # 5. Velocity correction (incompressible projection)
        dp_next_dx = central_difference_x(p_next, element_length)
        dp_next_dy = central_difference_y(p_next, element_length)

        u_next = (u_tentative - time_step_length / density * dp_next_dx) 
        v_next = (v_tentative - time_step_length / density * dp_next_dy)

        # 6. Boundary conditions for the final velocity
        u_next[0, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0
        u_next[-1, :] = horizontal_velocity
        
        v_next[0, :] = 0.0
        v_next[:, 0] = 0.0
        v_next[:, -1] = 0.0
        v_next[-1, :] = 0.0
        
        # 7. Advance in time
        u_prev = u_next
        v_prev = v_next
        p_prev = p_next

    return X, Y, u_next, v_next, p_next


# =====================================================================
# VISUALIZATION MODULE
# =====================================================================

def plot_flow_results(X: np.ndarray, Y: np.ndarray, u: np.ndarray, v: np.ndarray, p: np.ndarray) -> None:
    """
    Plots the fluid simulation results using matplotlib.

    Args:
        X (np.ndarray): Mesh X coordinates.
        Y (np.ndarray): Mesh Y coordinates.
        u (np.ndarray): Horizontal velocity field (X-axis).
        v (np.ndarray): Vertical velocity field (Y-axis).
        p (np.ndarray): Scalar pressure field.
    """
    plt.figure(figsize=(8, 6)) 
    
    # Plot the pressure field as a contour
    contour = plt.contourf(X, Y, p, cmap="viridis", alpha=0.8)
    plt.colorbar(contour, label="Pressure")

    # Plot the velocity vector field
    plt.quiver(X, Y, u, v, color="black")
    
    plt.title("Lid-Driven Cavity Flow Simulation")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


# =====================================================================
# MAIN EXECUTION (For direct testing)
# =====================================================================

if __name__ == "__main__":
    # 1. Call the solver (APIs can call this function and receive the JSON/Array data)
    X, Y, u_final, v_final, p_final = solve_navier_stokes(
        n_points=41,
        domain_size=1.0,
        n_iterations=500,
        time_step_length=0.0001,
        kinematic_viscosity=0.01,
        density=1.0,
        horizontal_velocity=1.0,
        n_pressure_poisson_iterations=50
    )
    
    # 2. Call the visualizer (decoupled from the calculations)
    plot_flow_results(X, Y, u_final, v_final, p_final)


