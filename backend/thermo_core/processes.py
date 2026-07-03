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

from utils.conversions import (
    convert_temperature_to_si,
    convert_pressure_to_si,
    convert_volume_to_si,
    convert_molar_volume_to_si,
    convert_energy_to_si,
    convert_molar_energy_to_si,
    convert_molar_entropy_to_si,
    convert_amount_to_moles
    )

"""
PR-SmartSolver Path Functions & Process Simulators.
Computes thermodynamic trajectories (Isobaric, Isothermal, Isochoric, Adiabatic-Isentropic)
for real fluids and multicomponent mixtures using the Peng-Robinson Cubic EOS.
All internal metadata and dictionary keys are strictly in English to match repository standards.
"""
def _calculate_residual_properties(
    press_val: float,
    press_unit: str,
    temp_val: float,
    temp_unit: str,
    vol_val: float,
    vol_unit: str,
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
        press_val (float): The numerical value of the absolute pressure.
        press_unit (str): The unit of the pressure as written in the prompt (e.g., 'Pa', 'atm', 'psi', 'bar', 'MPa').
        temp_val (float): The numerical value of the absolute temperature.
        temp_unit (str): The unit of the temperature as written in the prompt (e.g., 'C', 'F', 'K', 'R').
        vol_val (float): The numerical value of the molar volume.
        vol_unit (str): The unit of the molar volume as written in the prompt (e.g., 'm^3/mol', 'L/mol', 'cm^3/mol').
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
        ValueError: If pressure, temperature, or molar volume are less than or equal to zero after conversion to SI.
        ValueError: If the molar volume is less than or equal to the mixture co-volume (b_m), 
                    which violates physical boundaries and results in a math domain error 
                    (negative logarithm arguments in the S^R and H^R calculations).
        ValueError: If there is a dimension mismatch between fluids, fractions, and kij_matrix,
                    or if fractions are invalid (propagated from `calculate_pr_parameters`).
    """
    # Convert inputs to SI units (Pa for pressure, K for temperature, m^3/mol for molar volume)
    pressure = convert_pressure_to_si(press_val, press_unit)
    temperature = convert_temperature_to_si(temp_val, temp_unit)
    volume = convert_molar_volume_to_si(vol_val, vol_unit)
    
    if pressure <= 0 or temperature <= 0 or volume <= 0:
        raise ValueError("Pressure, temperature, and molar volume must be strictly positive after conversion to SI.")

    a_m, b_m, da_dT_m = calculate_pr_parameters(temperature, 'K', fluids, fractions, kij_matrix)
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
    press_val: float,
    press_unit: str,
    t_start_val: float,
    t_start_unit: str,
    t_end_val: float,
    t_end_unit: str,
    fluids: List[str],
    fractions: List[float],
    kij_matrix: np.ndarray,
    points: int = 20
) -> Dict[str, Any]:
    """
    Simulates an isobaric (constant pressure) thermodynamic heating or cooling trajectory.

    This function calculates the evolution of the compressibility factor (Z), 
    molar volume (V), and the relative changes in residual enthalpy (\Delta H^R), 
    residual entropy (\Delta S^R), and residual Gibbs free energy (\Delta G^R)
    across a defined temperature range.

    Args:
        press_val (float): The numerical value of the constant absolute pressure.
        press_unit (str): The unit of the pressure as written in the prompt (e.g., 'Pa', 'atm', 'psi', 'bar', 'MPa').
        t_start_val (float): The numerical value of the initial absolute temperature.
        t_start_unit (str): The unit of the initial temperature (e.g., 'C', 'F', 'K', 'R').
        t_end_val (float): The numerical value of the final absolute temperature.
        t_end_unit (str): The unit of the final temperature (e.g., 'C', 'F', 'K', 'R').
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
            - 'delta_H_J_mol' (List[float]): Change in residual enthalpy relative to initial state (J/mol).
            - 'delta_S_J_mol_K' (List[float]): Change in residual entropy relative to initial state (J/(mol*K)).
            - 'delta_G_J_mol' (List[float]): Change in residual Gibbs Free Energy relative to initial state (J/mol).

    Raises:
        ValueError: If `pressure`, `t_start`, or `t_end` are less than or equal to zero after conversion to SI.
        ValueError: Propagates EOS errors if physical boundaries are violated during 
                    the simulation (e.g., if volume attempts to drop below co-volume b_m).
    """
    # Convert inputs to SI units (Pa for pressure, K for temperature)
    pressure = convert_pressure_to_si(press_val, press_unit)
    t_start = convert_temperature_to_si(t_start_val, t_start_unit)
    t_end = convert_temperature_to_si(t_end_val, t_end_unit)

    if pressure <= 0 or t_start <= 0 or t_end <= 0:
        raise ValueError("Pressure and Temperatures must be strictly positive.")
        
    t_space = np.linspace(t_start, t_end, points)
    
    z_factors: List[float] = []
    volumes: List[float] = []
    delta_H: List[float] = []
    delta_S: List[float] = []
    delta_G: List[float] = []
    
    v_init = calculate_volume(pressure, 'Pa', t_start, 'K', fluids, fractions, kij_matrix)
    h_init, s_init, g_init = _calculate_residual_properties(pressure, 'Pa', t_start, 'K', v_init, 'm^3/mol', fluids, fractions, kij_matrix)
    
    for t in t_space:
        v = calculate_volume(pressure, 'Pa', t, 'K', fluids, fractions, kij_matrix)
        z = (pressure * v) / (UNIVERSAL_GAS_CONSTANT * t)
        h_res, s_res, g_res = _calculate_residual_properties(pressure, 'Pa', t, 'K', v, 'm^3/mol', fluids, fractions, kij_matrix)
        
        volumes.append(float(v))
        z_factors.append(float(z))
        delta_H.append(float(h_res - h_init))
        delta_S.append(float(s_res - s_init))
        delta_G.append(float(g_res - g_init))
        
    return {
        "process": "Isobaric",
        "x_axis_name": "Temperature (K)",
        "x_values": t_space.tolist(),
        "z_factors": z_factors,
        "volumes_m3_mol": volumes,
        "delta_H_J_mol": delta_H,
        "delta_S_J_mol_K": delta_S,
        "delta_G_J_mol": delta_G
    }


def simulate_isothermal_process(
    temp_val: float,
    temp_unit: str,
    p_start_val: float,
    p_start_unit: str,
    p_end_val: float,
    p_end_unit: str,
    fluids: List[str],
    fractions: List[float],
    kij_matrix: np.ndarray,
    points: int = 20
) -> Dict[str, Any]:
    """
    Simulates an isothermal (constant temperature) thermodynamic trajectory.

    This function calculates the evolution of the compressibility factor (Z), 
    molar volume (V), and the relative changes in residual enthalpy (\Delta H^R), 
    residual entropy (\Delta S^R), and residual Gibbs free energy (\Delta G^R)
    across a defined pressure range. 

    Tracking the shifts in these properties, particularly Gibbs Free Energy, is 
    valuable for evaluating phase stability and the spontaneity of processes 
    (e.g., in nanopores or separation units).

    Args:
        temp_val (float): The numerical value of the constant absolute temperature.
        temp_unit (str): The unit of the temperature as written in the prompt (e.g., 'C', 'F', 'K', 'R').
        p_start_val (float): The numerical value of the initial absolute pressure.
        p_start_unit (str): The unit of the initial pressure (e.g., 'Pa', 'atm', 'psi', 'bar', 'MPa').
        p_end_val (float): The numerical value of the final absolute pressure.
        p_end_unit (str): The unit of the final pressure (e.g., 'Pa', 'atm', 'psi', 'bar', 'MPa').
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
            - 'delta_H_J_mol' (List[float]): Change in residual enthalpy relative to initial state (J/mol).
            - 'delta_S_J_mol_K' (List[float]): Change in residual entropy relative to initial state (J/(mol*K)).
            - 'delta_G_J_mol' (List[float]): Change in residual Gibbs Free Energy relative to initial state (J/mol).

    Raises:
        ValueError: If `temperature`, `p_start`, or `p_end` are less than or equal to zero after conversion to SI.
        ValueError: Propagates EOS errors if physical boundaries are violated during 
                    the simulation (e.g., if volume attempts to drop below co-volume b_m 
                    at extreme pressures).
    """
    # Convert inputs to SI units (Pa for pressure, K for temperature)
    temperature = convert_temperature_to_si(temp_val, temp_unit)
    p_start = convert_pressure_to_si(p_start_val, p_start_unit)
    p_end = convert_pressure_to_si(p_end_val, p_end_unit)

    if temperature <= 0 or p_start <= 0 or p_end <= 0:
        raise ValueError("Temperature and Pressures must be strictly positive.")
        
    p_space = np.linspace(p_start, p_end, points)
    
    z_factors: List[float] = []
    volumes: List[float] = []
    delta_H: List[float] = []
    delta_S: List[float] = []
    delta_G: List[float] = []
    
    v_init = calculate_volume(p_start, 'Pa', temperature, 'K', fluids, fractions, kij_matrix)
    h_init, s_init, g_init = _calculate_residual_properties(p_start, 'Pa', temperature, 'K', v_init, 'm^3/mol', fluids, fractions, kij_matrix)
    
    for p in p_space:
        v = calculate_volume(p, 'Pa', temperature, 'K', fluids, fractions, kij_matrix)
        z = (p * v) / (UNIVERSAL_GAS_CONSTANT * temperature)
        h_res, s_res, g_res = _calculate_residual_properties(p, 'Pa', temperature, 'K', v, 'm^3/mol', fluids, fractions, kij_matrix)
        
        volumes.append(float(v))
        z_factors.append(float(z))
        delta_H.append(float(h_res - h_init))
        delta_S.append(float(s_res - s_init))
        delta_G.append(float(g_res - g_init))
        
    return {
        "process": "Isothermal",
        "x_axis_name": "Pressure (Pa)",
        "x_values": p_space.tolist(),
        "z_factors": z_factors,
        "volumes_m3_mol": volumes,
        "delta_H_J_mol": delta_H,
        "delta_S_J_mol_K": delta_S,
        "delta_G_J_mol": delta_G
    }

def simulate_isochoric_process(
    vol_val: float,
    vol_unit: str,
    t_start_val: float,
    t_start_unit: str,
    t_end_val: float,
    t_end_unit: str,
    fluids: List[str],
    fractions: List[float],
    kij_matrix: np.ndarray,
    points: int = 20
) -> Dict[str, Any]:
    """
    Simulates an isochoric (constant molar volume) thermodynamic heating or cooling trajectory.

    This function evaluates the evolution of absolute pressure (P), compressibility 
    factor (Z), and the relative changes in residual internal energy (\Delta U^R), 
    residual enthalpy (\Delta H^R), residual entropy (\Delta S^R), residual Gibbs 
    Free Energy (\Delta G^R), and residual Helmholtz Free Energy (\Delta A^R) across 
    a defined temperature range.

    Args:
        vol_val (float): The numerical value of the constant molar volume.
        vol_unit (str): The unit of the molar volume as written in the prompt (e.g., 'm^3/mol', 'L/mol', 'cm^3/mol').
        t_start_val (float): The numerical value of the initial absolute temperature.
        t_start_unit (str): The unit of the initial temperature (e.g., 'C', 'F', 'K', 'R').
        t_end_val (float): The numerical value of the final absolute temperature.
        t_end_unit (str): The unit of the final temperature (e.g., 'C', 'F', 'K', 'R').
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
            - 'pressures_pa' (List[float]): Calculated absolute pressures at each point (Pa).
            - 'delta_U_J_mol' (List[float]): Change in residual internal energy (J/mol).
            - 'delta_H_J_mol' (List[float]): Change in residual enthalpy (J/mol).
            - 'delta_S_J_mol_K' (List[float]): Change in residual entropy (J/(mol*K)).
            - 'delta_G_J_mol' (List[float]): Change in residual Gibbs Free Energy (J/mol).
            - 'delta_A_J_mol' (List[float]): Change in residual Helmholtz Free Energy (J/mol).

    Raises:
        ValueError: If `vol_val`, `t_start_val`, or `t_end_val` are less than or equal to zero after conversion to SI.
        ValueError: Propagates EOS errors if the constant volume is mathematically 
                    invalid (e.g., if v_m3_mol <= b_m).
    """
    # Convert inputs to SI units (m^3/mol for molar volume, K for temperature)
    v_m3_mol = convert_molar_volume_to_si(vol_val, vol_unit)
    t_start = convert_temperature_to_si(t_start_val, t_start_unit)
    t_end = convert_temperature_to_si(t_end_val, t_end_unit)

    if v_m3_mol <= 0 or t_start <= 0 or t_end <= 0:
        raise ValueError("Molar volume and Temperatures must be strictly positive.")
        
    t_space = np.linspace(t_start, t_end, points)
    
    z_factors: List[float] = []
    pressures: List[float] = []
    delta_U: List[float] = []
    delta_H: List[float] = []
    delta_S: List[float] = []
    delta_G: List[float] = []
    delta_A: List[float] = []
    
    # --- Initial State Calculations ---
    p_init = calculate_pressure(v_m3_mol, 'm^3/mol', t_start, 'K', fluids, fractions, kij_matrix)
    z_init = (p_init * v_m3_mol) / (UNIVERSAL_GAS_CONSTANT * t_start)
    h_init, s_init, g_init = _calculate_residual_properties(p_init, 'Pa', t_start, 'K', v_m3_mol, 'm^3/mol', fluids, fractions, kij_matrix)
    
    # Rigorous residual internal energy: U^R = H^R - RT(Z-1)
    u_init = h_init - UNIVERSAL_GAS_CONSTANT * t_start * (z_init - 1.0)
    # Residual Helmholtz free energy: A^R = U^R - TS^R
    a_init = u_init - t_start * s_init
    
    # --- Trajectory Calculations ---
    for t in t_space:
        p = calculate_pressure(v_m3_mol, 'm^3/mol', t, 'K', fluids, fractions, kij_matrix)
        z = (p * v_m3_mol) / (UNIVERSAL_GAS_CONSTANT * t)
        h_res, s_res, g_res = _calculate_residual_properties(p, 'Pa', t, 'K', v_m3_mol, 'm^3/mol', fluids, fractions, kij_matrix)
        
        u_res = h_res - UNIVERSAL_GAS_CONSTANT * t * (z - 1.0)
        a_res = u_res - t * s_res
        
        pressures.append(float(p))
        z_factors.append(float(z))
        
        # Calculate relative changes (Delta)
        delta_H.append(float(h_res - h_init))
        delta_S.append(float(s_res - s_init))
        delta_G.append(float(g_res - g_init))
        delta_U.append(float(u_res - u_init))
        delta_A.append(float(a_res - a_init))
        
    return {
        "process": "Isochoric",
        "x_axis_name": "Temperature (K)",
        "x_values": t_space.tolist(),
        "z_factors": z_factors,
        "volumes_m3_mol": [v_m3_mol] * points,
        "pressures_pa": pressures,
        "delta_U_J_mol": delta_U,
        "delta_H_J_mol": delta_H,
        "delta_S_J_mol_K": delta_S,
        "delta_G_J_mol": delta_G,
        "delta_A_J_mol": delta_A
    }

def simulate_adiabatic_process(
    p_start_val: float,
    p_start_unit: str,
    p_end_val: float,
    p_end_unit: str,
    t_start_val: float,
    t_start_unit: str,
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

    It comprehensively tracks the evolution of the compressibility factor (Z), 
    molar volume (V), steady-flow work, and all relative changes in residual 
    properties (\Delta H^R, \Delta S^R, \Delta G^R, \Delta U^R, \Delta A^R).

    Args:
        p_start_val (float): The numerical value of the initial absolute pressure.
        p_start_unit (str): The unit of the initial pressure (e.g., 'Pa', 'atm', 'psi', 'bar', 'MPa').
        p_end_val (float): The numerical value of the final absolute pressure.
        p_end_unit (str): The unit of the final pressure (e.g., 'Pa', 'atm', 'psi', 'bar', 'MPa').
        t_start_val (float): The numerical value of the initial absolute temperature.
        t_start_unit (str): The unit of the initial temperature (e.g., 'C', 'F', 'K', 'R').
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
            - 'work_produced_J_mol' (List[float]): Cumulative steady-flow work (J/mol).
            - 'delta_H_J_mol' (List[float]): Change in residual enthalpy (J/mol).
            - 'delta_S_J_mol_K' (List[float]): Change in residual entropy (should be ~0) (J/(mol*K)).
            - 'delta_G_J_mol' (List[float]): Change in residual Gibbs Free Energy (J/mol).
            - 'delta_U_J_mol' (List[float]): Change in residual internal energy (J/mol).
            - 'delta_A_J_mol' (List[float]): Change in residual Helmholtz Free Energy (J/mol).

    Raises:
        ValueError: If `p_start`, `p_end`, or `t_start` are less than or equal to zero after conversion to SI.
        ValueError: If the numerical Secant solver fails to converge on a valid 
                    isentropic temperature for any given pressure point.
    """
    # Convert inputs to SI units (Pa for pressure, K for temperature)
    p_start = convert_pressure_to_si(p_start_val, p_start_unit)
    p_end = convert_pressure_to_si(p_end_val, p_end_unit)
    t_start = convert_temperature_to_si(t_start_val, t_start_unit)

    if p_start <= 0 or p_end <= 0 or t_start <= 0:
        raise ValueError("Pressures and Initial Temperature must be strictly positive.")
        
    p_space = np.linspace(p_start, p_end, points)
    
    # --- Initial State Calculations ---
    v_init = calculate_volume(p_start, 'Pa', t_start, 'K', fluids, fractions, kij_matrix)
    z_init = (p_start * v_init) / (UNIVERSAL_GAS_CONSTANT * t_start)
    h_init, s_target, g_init = _calculate_residual_properties(p_start, 'Pa', t_start, 'K', v_init, 'm^3/mol', fluids, fractions, kij_matrix)
    
    u_init = h_init - UNIVERSAL_GAS_CONSTANT * t_start * (z_init - 1.0)
    a_init = u_init - t_start * s_target
    
    z_factors: List[float] = []
    volumes: List[float] = []
    temperatures: List[float] = []
    work_produced: List[float] = []
    
    delta_H: List[float] = []
    delta_S: List[float] = []
    delta_G: List[float] = []
    delta_U: List[float] = []
    delta_A: List[float] = []
    
    t_guess = t_start
    
    # --- Trajectory Calculations ---
    for p in p_space:
        def entropy_objective(t_est: float) -> float:
            if t_est <= 0:
                return 1e9
            try:
                v_est = calculate_volume(p, 'Pa', t_est, 'K', fluids, fractions, kij_matrix)
                _, s_est, _ = _calculate_residual_properties(p, 'Pa', t_est, 'K', v_est, 'm^3/mol', fluids, fractions, kij_matrix)
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
        v_real = calculate_volume(p, 'Pa', t_real, 'K', fluids, fractions, kij_matrix)
        z_real = (p * v_real) / (UNIVERSAL_GAS_CONSTANT * t_real)
        h_real, s_real, g_real = _calculate_residual_properties(p, 'Pa', t_real, 'K', v_real, 'm^3/mol', fluids, fractions, kij_matrix)
        
        u_real = h_real - UNIVERSAL_GAS_CONSTANT * t_real * (z_real - 1.0)
        a_real = u_real - t_real * s_real
        
        temperatures.append(t_real)
        volumes.append(v_real)
        z_factors.append(z_real)
        work_produced.append(float(h_init - h_real))
        
        delta_H.append(float(h_real - h_init))
        delta_S.append(float(s_real - s_target))
        delta_G.append(float(g_real - g_init))
        delta_U.append(float(u_real - u_init))
        delta_A.append(float(a_real - a_init))
        
        t_guess = t_real
        
    return {
        "process": "Adiabatic",
        "x_axis_name": "Pressure (Pa)",
        "x_values": p_space.tolist(),
        "z_factors": z_factors,
        "volumes_m3_mol": volumes,
        "temperatures_K": temperatures,
        "work_produced_J_mol": work_produced,
        "delta_H_J_mol": delta_H,
        "delta_S_J_mol_K": delta_S,
        "delta_G_J_mol": delta_G,
        "delta_U_J_mol": delta_U,
        "delta_A_J_mol": delta_A
    }