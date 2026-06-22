import numpy as np
import scipy.optimize as opt
from typing import Dict, List, Any, Tuple

from .eos import (
    UNIVERSAL_GAS_CONSTANT,
    calculate_volume,
    calculate_pressure,
    calculate_pr_parameters,
    calculate_compressibility_factor
)

"""
PR-SmartSolver Path Functions & Process Simulators.
Computes thermodynamic trajectories (Isobaric, Isothermal, Isochoric, Adiabatic-Isentropic)
for real fluids and multicomponent mixtures using the Peng-Robinson Cubic EOS.
All internal metadata and dictionary keys are strictly in English to match repository standards.
"""
def _calculate_residual_properties(
    pressure: float,
    temperature: float,
    volume: float,
    fluids: List[str],
    fractions: List[float],
    kij_matrix: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculates the exact analytical residual Enthalpy (H^R), Entropy (S^R), and 
    Gibbs Free Energy (G^R) for a single component or multicomponent mixture.

    This internal helper function applies the Peng-Robinson Equation of State (EOS) 
    analytical derivations to compute the deviation of the thermodynamic properties 
    from an ideal gas behavior at the same temperature and pressure.

    Args:
        pressure (float): Absolute pressure of the system in Pascals (Pa).
        temperature (float): Absolute temperature of the system in Kelvin (K).
        volume (float): Molar volume of the system in cubic meters per mole (m^3/mol).
        fluids (List[str]): List of fluid names recognized by CoolProp.
        fractions (List[float]): Molar fractions of each component in the mixture.
        kij_matrix (np.ndarray): A 2D symmetric NumPy array of shape (n, n) representing 
                                 the binary interaction parameters (k_ij).

    Returns:
        Tuple[float, float, float]: A tuple containing the residual properties:
            - h_res (float): Residual Enthalpy (H^R) in Joules per mole (J/mol).
            - s_res (float): Residual Entropy (S^R) in Joules per mole-Kelvin (J/(mol*K)).
            - g_res (float): Residual Gibbs Free Energy (G^R) in Joules per mole (J/mol).

    Raises:
        ValueError: If pressure, temperature, or molar volume are less than or equal to zero.
        ValueError: If the molar volume is less than or equal to the mixture co-volume (b_m), 
                    which violates physical boundaries and results in a math domain error 
                    (negative logarithm arguments in the S^R and H^R calculations).
        ValueError: If there is a dimension mismatch between fluids, fractions, and kij_matrix,
                    or if fractions are invalid (propagated from `calculate_pr_parameters`).
    """
    a_m, b_m, da_dT_m = calculate_pr_parameters(temperature, fluids, fractions, kij_matrix)
    Z = (pressure * volume) / (UNIVERSAL_GAS_CONSTANT * temperature)
    
    sqrt2 = np.sqrt(2.0)
    ln_term = np.log((volume + b_m * (1.0 + sqrt2)) / (volume + b_m * (1.0 - sqrt2)))
    
    # Enthalpy Residual (H^R) analytical expression
    h_res = (UNIVERSAL_GAS_CONSTANT * temperature * (Z - 1.0) + 
             (temperature * da_dT_m - a_m) / (2.0 * sqrt2 * b_m) * ln_term)
    
    # Entropy Residual (S^R) analytical expression
    s_res = (UNIVERSAL_GAS_CONSTANT * np.log(Z - (b_m * pressure / (UNIVERSAL_GAS_CONSTANT * temperature))) + 
             da_dT_m / (2.0 * sqrt2 * b_m) * ln_term)
    
    # Gibbs Free Energy Residual (G^R = H^R - T*S^R)
    g_res = h_res - temperature * s_res
    
    return h_res, s_res, g_res


def simulate_isobaric_process(
    pressure: float,
    t_start: float,
    t_end: float,
    fluids: List[str],
    fractions: List[float],
    kij_matrix: np.ndarray,
    points: int = 20
) -> Dict[str, Any]:
    """
    Simulates an isobaric (constant pressure) thermodynamic heating or cooling trajectory.

    This function calculates the evolution of the compressibility factor (Z), 
    molar volume (V), and the relative change in residual enthalpy (\Delta H^R) 
    across a defined temperature range. The simulation provides an array of states 
    that can be directly plotted or analyzed by downstream tools.

    Args:
        pressure (float): Constant absolute pressure of the system in Pascals (Pa).
        t_start (float): Initial absolute temperature in Kelvin (K).
        t_end (float): Final absolute temperature in Kelvin (K).
        fluids (List[str]): List of fluid names recognized by CoolProp.
        fractions (List[float]): Molar fractions of each component in the mixture.
        kij_matrix (np.ndarray): A 2D symmetric NumPy array of shape (n, n) representing 
                                 the binary interaction parameters (k_ij).
        points (int, optional): Number of discrete temperature points to simulate 
                                along the trajectory. Defaults to 20.

    Returns:
        Dict[str, Any]: A dictionary containing the simulation trajectory arrays:
            - 'process' (str): The name of the simulated process ("Isobaric").
            - 'x_axis_name' (str): Label for the independent variable ("Temperature (K)").
            - 'x_values' (List[float]): The discrete temperature points evaluated (K).
            - 'z_factors' (List[float]): Compressibility factors (Z) at each evaluated point.
            - 'volumes_m3_mol' (List[float]): Molar volumes at each point (m^3/mol).
            - 'delta_H_J_mol' (List[float]): Change in residual enthalpy relative to 
                                             the initial state (t_start) in J/mol.

    Raises:
        ValueError: If `pressure`, `t_start`, or `t_end` are less than or equal to zero.
        ValueError: Propagates EOS errors if physical boundaries are violated during 
                    the simulation (e.g., if volume attempts to drop below co-volume b_m).
    """
    if pressure <= 0 or t_start <= 0 or t_end <= 0:
        raise ValueError("Pressure and Temperatures must be strictly positive.")
        
    t_space = np.linspace(t_start, t_end, points)
    
    z_factors: List[float] = []
    volumes: List[float] = []
    delta_H: List[float] = []
    
    v_init = calculate_volume(pressure, t_start, fluids, fractions, kij_matrix)
    h_init, _, _ = _calculate_residual_properties(pressure, t_start, v_init, fluids, fractions, kij_matrix)
    
    for t in t_space:
        v = calculate_volume(pressure, t, fluids, fractions, kij_matrix)
        z = (pressure * v) / (UNIVERSAL_GAS_CONSTANT * t)
        h_res, _, _ = _calculate_residual_properties(pressure, t, v, fluids, fractions, kij_matrix)
        
        volumes.append(float(v))
        z_factors.append(float(z))
        delta_H.append(float(h_res - h_init))
        
    return {
        "process": "Isobaric",
        "x_axis_name": "Temperature (K)",
        "x_values": t_space.tolist(),
        "z_factors": z_factors,
        "volumes_m3_mol": volumes,
        "delta_H_J_mol": delta_H
    }

def simulate_isothermal_process(
    temperature: float,
    p_start: float,
    p_end: float,
    fluids: List[str],
    fractions: List[float],
    kij_matrix: np.ndarray,
    points: int = 20
) -> Dict[str, Any]:
    """
    Simulates an isothermal (constant temperature) thermodynamic trajectory.

    This function calculates the evolution of the compressibility factor (Z), 
    molar volume (V), and the relative change in residual Gibbs Free Energy (\Delta G^R) 
    across a defined pressure range. Tracking the shifts in Gibbs Free Energy is 
    particularly valuable for evaluating phase stability and the spontaneity of 
    processes (e.g., in nanopores or separation units).

    Args:
        temperature (float): Constant absolute temperature of the system in Kelvin (K).
        p_start (float): Initial absolute pressure of the system in Pascals (Pa).
        p_end (float): Final absolute pressure of the system in Pascals (Pa).
        fluids (List[str]): List of fluid names recognized by CoolProp.
        fractions (List[float]): Molar fractions of each component in the mixture.
        kij_matrix (np.ndarray): A 2D symmetric NumPy array of shape (n, n) representing 
                                 the binary interaction parameters (k_ij).
        points (int, optional): Number of discrete pressure points to simulate 
                                along the trajectory. Defaults to 20.

    Returns:
        Dict[str, Any]: A dictionary containing the simulation trajectory arrays:
            - 'process' (str): The name of the simulated process ("Isothermal").
            - 'x_axis_name' (str): Label for the independent variable ("Pressure (Pa)").
            - 'x_values' (List[float]): The discrete pressure points evaluated (Pa).
            - 'z_factors' (List[float]): Compressibility factors (Z) at each evaluated point.
            - 'volumes_m3_mol' (List[float]): Molar volumes at each point (m^3/mol).
            - 'delta_G_J_mol' (List[float]): Change in residual Gibbs Free Energy relative to 
                                             the initial state (p_start) in J/mol.

    Raises:
        ValueError: If `temperature`, `p_start`, or `p_end` are less than or equal to zero.
        ValueError: Propagates EOS errors if physical boundaries are violated during 
                    the simulation (e.g., if volume attempts to drop below co-volume b_m 
                    at extreme pressures).
    """
    if temperature <= 0 or p_start <= 0 or p_end <= 0:
        raise ValueError("Temperature and Pressures must be strictly positive.")
        
    p_space = np.linspace(p_start, p_end, points)
    
    z_factors: List[float] = []
    volumes: List[float] = []
    delta_G: List[float] = []
    
    v_init = calculate_volume(p_start, temperature, fluids, fractions, kij_matrix)
    _, _, g_init = _calculate_residual_properties(p_start, temperature, v_init, fluids, fractions, kij_matrix)
    
    for p in p_space:
        v = calculate_volume(p, temperature, fluids, fractions, kij_matrix)
        z = (p * v) / (UNIVERSAL_GAS_CONSTANT * temperature)
        _, _, g_res = _calculate_residual_properties(p, temperature, v, fluids, fractions, kij_matrix)
        
        volumes.append(float(v))
        z_factors.append(float(z))
        delta_G.append(float(g_res - g_init))
        
    return {
        "process": "Isothermal",
        "x_axis_name": "Pressure (Pa)",
        "x_values": p_space.tolist(),
        "z_factors": z_factors,
        "volumes_m3_mol": volumes,
        "delta_G_J_mol": delta_G
    }

def simulate_isochoric_process(
    v_m3_mol: float,
    t_start: float,
    t_end: float,
    fluids: List[str],
    fractions: List[float],
    kij_matrix: np.ndarray,
    points: int = 20
) -> Dict[str, Any]:
    """
    Simulates an isochoric (constant molar volume) thermodynamic heating or cooling trajectory.

    This function evaluates the evolution of absolute pressure (P), compressibility 
    factor (Z), and the relative change in internal energy (\Delta U) across a 
    defined temperature range while keeping the molar volume strictly constant. 
    The resulting arrays are formatted for downstream analysis or direct plotting.

    Args:
        v_m3_mol (float): Constant molar volume of the system in cubic meters per mole (m^3/mol).
        t_start (float): Initial absolute temperature in Kelvin (K).
        t_end (float): Final absolute temperature in Kelvin (K).
        fluids (List[str]): List of fluid names recognized by CoolProp.
        fractions (List[float]): Molar fractions of each component in the mixture.
        kij_matrix (np.ndarray): A 2D symmetric NumPy array of shape (n, n) representing 
                                 the binary interaction parameters (k_ij).
        points (int, optional): Number of discrete temperature points to simulate 
                                along the trajectory. Defaults to 20.

    Returns:
        Dict[str, Any]: A dictionary containing the simulation trajectory arrays:
            - 'process' (str): The name of the simulated process ("Isochoric").
            - 'x_axis_name' (str): Label for the independent variable ("Temperature (K)").
            - 'x_values' (List[float]): The discrete temperature points evaluated (K).
            - 'z_factors' (List[float]): Compressibility factors (Z) at each evaluated point.
            - 'volumes_m3_mol' (List[float]): Constant molar volumes repeated for each point (m^3/mol).
            - 'delta_U_J_mol' (List[float]): Change in internal energy relative to 
                                             the initial state (t_start) in Joules per mole (J/mol).
            - 'pressures_pa' (List[float]): Calculated absolute pressures at each point (Pa).

    Raises:
        ValueError: If `v_m3_mol`, `t_start`, or `t_end` are less than or equal to zero.
        ValueError: Propagates EOS errors if the constant volume is mathematically 
                    invalid (e.g., if v_m3_mol <= b_m).
    """
    if v_m3_mol <= 0 or t_start <= 0 or t_end <= 0:
        raise ValueError("Molar volume and Temperatures must be strictly positive.")
        
    t_space = np.linspace(t_start, t_end, points)
    
    z_factors: List[float] = []
    pressures: List[float] = []
    delta_U: List[float] = []
    
    p_init = calculate_pressure(v_m3_mol, t_start, fluids, fractions, kij_matrix)
    h_init, _, _ = _calculate_residual_properties(p_init, t_start, v_m3_mol, fluids, fractions, kij_matrix)
    u_init = h_init - (p_init * v_m3_mol)
    
    for t in t_space:
        p = calculate_pressure(v_m3_mol, t, fluids, fractions, kij_matrix)
        z = (p * v_m3_mol) / (UNIVERSAL_GAS_CONSTANT * t)
        h_res, _, _ = _calculate_residual_properties(p, t, v_m3_mol, fluids, fractions, kij_matrix)
        u_res = h_res - (p * v_m3_mol)
        
        pressures.append(float(p))
        z_factors.append(float(z))
        delta_U.append(float(u_res - u_init))
        
    return {
        "process": "Isochoric",
        "x_axis_name": "Temperature (K)",
        "x_values": t_space.tolist(),
        "z_factors": z_factors,
        "volumes_m3_mol": [v_m3_mol] * points,
        "delta_U_J_mol": delta_U,
        "pressures_pa": pressures
    }


def simulate_adiabatic_process(
    p_start: float,
    p_end: float,
    t_start: float,
    fluids: List[str],
    fractions: List[float],
    kij_matrix: np.ndarray,
    points: int = 20
) -> Dict[str, Any]:
    """
    Simulates a reversible adiabatic (isentropic) thermodynamic trajectory.

    This function calculates the temperature profile of a fluid undergoing an 
    adiabatic compression or expansion where the entropy change is zero (\Delta S = 0). 
    It uses a numerical root-finding algorithm (Secant method) to iteratively solve 
    for the exact temperature at each pressure step that maintains the initial 
    residual entropy.

    It also calculates the steady-flow adiabatic work produced or consumed 
    (W = \Delta H) based on the residual enthalpy shifts.

    Args:
        p_start (float): Initial absolute pressure of the system in Pascals (Pa).
        p_end (float): Final absolute pressure of the system in Pascals (Pa).
        t_start (float): Initial absolute temperature in Kelvin (K).
        fluids (List[str]): List of fluid names recognized by CoolProp.
        fractions (List[float]): Molar fractions of each component in the mixture.
        kij_matrix (np.ndarray): A 2D symmetric NumPy array representing the 
                                 binary interaction parameters (k_ij).
        points (int, optional): Number of discrete pressure points to simulate 
                                along the trajectory. Defaults to 20.

    Returns:
        Dict[str, Any]: A dictionary containing the simulation trajectory arrays:
            - 'process' (str): The name of the simulated process ("Adiabatic").
            - 'x_axis_name' (str): Label for the independent variable ("Pressure (Pa)").
            - 'x_values' (List[float]): The discrete pressure points evaluated (Pa).
            - 'z_factors' (List[float]): Compressibility factors (Z) at each evaluated point.
            - 'volumes_m3_mol' (List[float]): Molar volumes at each point (m^3/mol).
            - 'temperatures_K' (List[float]): Calculated isentropic temperatures (K).
            - 'work_produced_J_mol' (List[float]): Cumulative steady-flow work produced 
                                                   or consumed (J/mol), calculated as (H_init - H_real).

    Raises:
        ValueError: If `p_start`, `p_end`, or `t_start` are less than or equal to zero.
        ValueError: If the numerical Secant solver fails to converge on a valid 
                    isentropic temperature for any given pressure point.
    """
    if p_start <= 0 or p_end <= 0 or t_start <= 0:
        raise ValueError("Pressures and Initial Temperature must be strictly positive.")
        
    p_space = np.linspace(p_start, p_end, points)
    
    v_init = calculate_volume(p_start, t_start, fluids, fractions, kij_matrix)
    h_init, s_target, _ = _calculate_residual_properties(p_start, t_start, v_init, fluids, fractions, kij_matrix)
    
    z_factors: List[float] = []
    volumes: List[float] = []
    temperatures: List[float] = []
    work_produced: List[float] = []
    
    t_guess = t_start
    
    for p in p_space:
        def entropy_objective(t_est: float) -> float:
            if t_est <= 0:
                return 1e9
            try:
                v_est = calculate_volume(p, t_est, fluids, fractions, kij_matrix)
                _, s_est, _ = _calculate_residual_properties(p, t_est, v_est, fluids, fractions, kij_matrix)
                return s_est - s_target
            except ValueError:
                return 1e9

        sol = opt.root_scalar(
            entropy_objective, 
            x0=t_guess, 
            x1=t_guess + 1.0, 
            method='secant', 
            xtol=1e-5, 
            maxiter=50
        )
        
        if not sol.converged:
            raise ValueError(f"Isentropic convergence failure at pressure P={p:.2e} Pa.")
            
        t_real = float(sol.root)
        v_real = calculate_volume(p, t_real, fluids, fractions, kij_matrix)
        z_real = (p * v_real) / (UNIVERSAL_GAS_CONSTANT * t_real)
        h_real, _, _ = _calculate_residual_properties(p, t_real, v_real, fluids, fractions, kij_matrix)
        
        temperatures.append(t_real)
        volumes.append(v_real)
        z_factors.append(z_real)
        work_produced.append(float(h_init - h_real))
        
        t_guess = t_real
        
    return {
        "process": "Adiabatic",
        "x_axis_name": "Pressure (Pa)",
        "x_values": p_space.tolist(),
        "z_factors": z_factors,
        "volumes_m3_mol": volumes,
        "temperatures_K": temperatures,
        "work_produced_J_mol": work_produced
    }