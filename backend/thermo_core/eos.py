import numpy as np
import scipy.optimize as opt
import CoolProp.CoolProp as CP
from typing import Tuple, List

# Universal gas constant in J/(mol*K)
UNIVERSAL_GAS_CONSTANT = 8.314462618

def get_gas_properties(fluids: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the critical properties of a list of fluids using CoolProp.

    Args:
        fluids (List[str]): List of fluid names (e.g., ['Methane', 'Ethane']).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays for Tc (K), Pc (Pa), and omega.
    """
    tc_list, pc_list, omega_list = [], [], []
    for fluid in fluids:
        try:
            tc_list.append(float(CP.PropsSI("Tcrit", fluid)))
            pc_list.append(float(CP.PropsSI("pcrit", fluid)))
            omega_list.append(float(CP.PropsSI("acentric", fluid)))
        except ValueError as e:
            raise ValueError(f"Fluid '{fluid}' is not available in CoolProp. Original error: {e}")
            
    return np.array(tc_list), np.array(pc_list), np.array(omega_list)


def calculate_pr_parameters(
    temperature: float, 
    fluids: List[str], 
    fractions: List[float], 
    kij_matrix: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculates the Peng-Robinson parameters (a, b) and the temperature derivative (da/dT)
    for a multicomponent mixture using Van der Waals mixing rules.

    Args:
        temperature (float): Absolute temperature of the system in Kelvin (K).
        fluids (List[str]): List of fluid names as recognized by CoolProp.
        fractions (List[float]): Molar fractions of each component in the mixture.
        kij_matrix (np.ndarray): A 2D symmetric NumPy array of shape (n, n) representing 
                                 the binary interaction parameters (k_ij) between components.

    Returns:
        Tuple[float, float, float]: A tuple containing:
            - a_m (float): The mixture attraction parameter.
            - b_m (float): The mixture co-volume parameter.
            - da_dT_m (float): The analytical derivative of a_m with respect to temperature.

    Raises:
        ValueError: If temperature is less than or equal to zero.
        ValueError: If the list of fluids is empty.
        ValueError: If the length of fractions does not match the number of fluids.
        ValueError: If the kij_matrix is not a square matrix of size (n, n).
        ValueError: If any molar fraction is negative or if they do not sum to 1.0.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be strictly positive (T > 0).")
        
    n = len(fluids)
    if n == 0:
        raise ValueError("The list of fluids cannot be empty.")
        
    x = np.array(fractions)
    if len(x) != n:
        raise ValueError(f"Dimension mismatch: expected {n} fractions, got {len(x)}.")
        
    if kij_matrix.shape != (n, n):
        raise ValueError(f"Dimension mismatch: kij_matrix must be of shape ({n}, {n}).")
        
    if np.any(x < 0) or not np.isclose(np.sum(x), 1.0):
        raise ValueError("Molar fractions must be non-negative and sum exactly to 1.0.")

    tc, pc, omega = get_gas_properties(fluids)

    # 1. Pure component parameters (Vectorized calculations)
    tr = temperature / tc
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + kappa * (1 - np.sqrt(tr))) ** 2

    a_ind = 0.45724 * (UNIVERSAL_GAS_CONSTANT**2 * tc**2) / pc * alpha
    b_ind = 0.07780 * (UNIVERSAL_GAS_CONSTANT * tc) / pc
    
    # Analytical derivative of alpha and 'a' for pure components
    dalpha_dT = -kappa * (1 + kappa * (1 - np.sqrt(tr))) / (tc * np.sqrt(tr))
    da_dT_ind = 0.45724 * (UNIVERSAL_GAS_CONSTANT**2 * tc**2) / pc * dalpha_dT

    # 2. Van der Waals Mixing Rules
    b_m = np.sum(x * b_ind)

    a_m = 0.0
    da_dT_m = 0.0
    for i in range(n):
        for j in range(n):
            # Attraction parameter a_ij
            a_ij = np.sqrt(a_ind[i] * a_ind[j]) * (1.0 - kij_matrix[i, j])
            a_m += x[i] * x[j] * a_ij
            
            # Cross derivative da_ij/dT
            term1 = da_dT_ind[i] * np.sqrt(a_ind[j] / a_ind[i]) if a_ind[i] > 0 else 0
            term2 = da_dT_ind[j] * np.sqrt(a_ind[i] / a_ind[j]) if a_ind[j] > 0 else 0
            da_ij_dT = 0.5 * (term1 + term2) * (1.0 - kij_matrix[i, j])
            da_dT_m += x[i] * x[j] * da_ij_dT

    return a_m, b_m, da_dT_m


def calculate_pressure(
    volume: float, 
    temperature: float, 
    fluids: List[str], 
    fractions: List[float], 
    kij_matrix: np.ndarray
) -> float:
    """
    Calculates the pressure of a pure fluid or a multicomponent mixture 
    using the Peng-Robinson Equation of State.

    Args:
        volume (float): Molar volume of the system in m^3/mol.
        temperature (float): Absolute temperature of the system in Kelvin (K).
        fluids (List[str]): List of fluid names recognized by CoolProp.
        fractions (List[float]): Molar fractions of each component in the mixture.
        kij_matrix (np.ndarray): A 2D symmetric NumPy array representing the 
                                 binary interaction parameters (k_ij).

    Returns:
        float: Calculated pressure in Pascals (Pa).

    Raises:
        ValueError: If molar volume or temperature is less than or equal to zero.
        ValueError: If the molar volume is less than or equal to the mixture co-volume (b_m),
                    which is physically impossible and causes mathematical singularities.
    """
    if volume <= 0:
        raise ValueError("Molar volume must be strictly positive (V > 0).")
    if temperature <= 0:
        raise ValueError("Temperature must be strictly positive (T > 0).")

    # Fetch mixture parameters
    a_m, b_m, _ = calculate_pr_parameters(temperature, fluids, fractions, kij_matrix)
    
    # Critical Physical Validation: Volume must be greater than the excluded volume
    if volume <= b_m:
        raise ValueError(
            f"Physically impossible state: Molar volume (V = {volume:.5e}) "
            f"must be strictly greater than the mixture co-volume (b_m = {b_m:.5e})."
        )

    # Peng-Robinson Equation
    term1 = (UNIVERSAL_GAS_CONSTANT * temperature) / (volume - b_m)
    term2 = a_m / (volume * (volume + b_m) + b_m * (volume - b_m))
    
    return term1 - term2


def calculate_volume(
    pressure: float, 
    temperature: float, 
    fluids: List[str], 
    fractions: List[float], 
    kij_matrix: np.ndarray
) -> float:
    """
    Calculates the molar volume of a gas/vapor mixture given pressure and temperature.
    
    This function solves the cubic Peng-Robinson equation for the compressibility 
    factor (Z) and extracts the maximum real root, which mathematically corresponds 
    to the stable gas, vapor, or supercritical phase.

    Args:
        pressure (float): Absolute pressure of the system in Pascals (Pa).
        temperature (float): Absolute temperature of the system in Kelvin (K).
        fluids (List[str]): List of fluid names recognized by CoolProp.
        fractions (List[float]): Molar fractions of each component in the mixture.
        kij_matrix (np.ndarray): A 2D symmetric NumPy array representing the 
                                 binary interaction parameters (k_ij).

    Returns:
        float: Calculated molar volume in m^3/mol.

    Raises:
        ValueError: If pressure or temperature is less than or equal to zero.
        ValueError: If no valid positive real roots are found for Z.
        ValueError: If the calculated volume is physically impossible (V <= b_m).
    """
    if pressure <= 0:
        raise ValueError("Pressure must be strictly positive (P > 0).")
    if temperature <= 0:
        raise ValueError("Temperature must be strictly positive (T > 0).")

    # Fetch mixture parameters
    a_m, b_m, _ = calculate_pr_parameters(temperature, fluids, fractions, kij_matrix)
    
    # Solve for Z (maximum real root for vapor/gas phase)
    z_max = calculate_compressibility_factor(pressure, temperature, a_m, b_m)
    
    # Calculate volume using the real gas law
    molar_volume = z_max * UNIVERSAL_GAS_CONSTANT * temperature / pressure

    # Physical reality check: Volume must exceed the excluded physical space of molecules
    if molar_volume <= b_m:
        raise ValueError(
            f"Calculated state is physically invalid: Molar volume (V = {molar_volume:.5e}) "
            f"is less than or equal to the mixture co-volume (b_m = {b_m:.5e})."
        )

    return molar_volume


def calculate_temperature(
    pressure: float, 
    volume: float, 
    fluids: List[str], 
    fractions: List[float], 
    kij_matrix: np.ndarray
) -> float:
    """
    Calculates the absolute temperature of a gas/vapor mixture given its pressure 
    and molar volume by finding the root of the Peng-Robinson pressure function.

    This function utilizes Brent's method (brentq) to solve for T such that:
    P_calc(T, V, x) - P_target = 0. 
    
    To ensure convergence, it establishes a dynamic search bracket based on the 
    mixture's pseudo-critical temperature (Kay's rule).

    Args:
        pressure (float): Target absolute pressure in Pascals (Pa).
        volume (float): Molar volume of the system in m^3/mol.
        fluids (List[str]): List of fluid names recognized by CoolProp.
        fractions (List[float]): Molar fractions of each component in the mixture.
        kij_matrix (np.ndarray): A 2D symmetric NumPy array representing the 
                                 binary interaction parameters (k_ij).

    Returns:
        float: Calculated absolute temperature in Kelvin (K).

    Raises:
        ValueError: If pressure or volume is less than or equal to zero.
        ValueError: If the calculated state implies a volume smaller than the 
                    mixture's co-volume (b_m).
        ValueError: If the root-finding algorithm fails to converge within the 
                    established temperature bracket.
    """
    if pressure <= 0:
        raise ValueError("Pressure must be strictly positive (P > 0).")
    if volume <= 0:
        raise ValueError("Molar volume must be strictly positive (V > 0).")

    # Estimate mixture pseudo-critical temperature using Kay's rule
    # This provides a physically sound anchor for the solver's search bracket
    tc, _, _ = get_gas_properties(fluids)
    tc_mix_approx = np.sum(np.array(fractions) * tc)

    def objective_function(t_est: float) -> float:
        # calculate_pressure will naturally raise a ValueError if volume <= b_m
        return calculate_pressure(volume, t_est, fluids, fractions, kij_matrix) - pressure

    try:
        # Brent's method requires the root to be bracketed between f(a) and f(b) with opposite signs
        solution = opt.root_scalar(
            objective_function, 
            bracket=[tc_mix_approx * 0.1, tc_mix_approx * 5.0], 
            method="brentq"
        )
        
        if not solution.converged:
            raise ValueError(
                f"Brent's method failed to converge for P={pressure:.2e} Pa and V={volume:.5e} m^3/mol."
            )
            
        return float(solution.root)
        
    except ValueError as e:
        # Catches both root_scalar convergence issues and physical violations from calculate_pressure
        raise ValueError(f"Failed to resolve a valid temperature. Reason: {e}")


def calculate_compressibility_factor(
    pressure: float, 
    temperature: float, 
    a: float, 
    b: float
) -> float:
    """
    Solves the cubic Peng-Robinson polynomial to find the compressibility factor (Z).

    This function constructs the dimensionless parameters A and B, solves the 
    cubic equation of state, and extracts the maximum positive real root. 
    The maximum root mathematically corresponds to the stable gas, vapor, 
    or supercritical phase.

    Args:
        pressure (float): Absolute pressure of the system in Pascals (Pa).
        temperature (float): Absolute temperature of the system in Kelvin (K).
        a (float): The Peng-Robinson attraction parameter (pure or mixture).
        b (float): The Peng-Robinson co-volume parameter (pure or mixture).

    Returns:
        float: The compressibility factor (Z), dimensionless.

    Raises:
        ValueError: If pressure or temperature is less than or equal to zero.
        ValueError: If no valid positive real roots are found.
    """
    if pressure <= 0:
        raise ValueError("Pressure must be strictly positive (P > 0).")
    if temperature <= 0:
        raise ValueError("Temperature must be strictly positive (T > 0).")

    # Dimensionless coefficients for the cubic equation
    A_coeff = (a * pressure) / (UNIVERSAL_GAS_CONSTANT**2 * temperature**2)
    B_coeff = (b * pressure) / (UNIVERSAL_GAS_CONSTANT * temperature)

    # Cubic equation coefficients: Z^3 + c2*Z^2 + c1*Z + c0 = 0
    c2 = -(1 - B_coeff)
    c1 = A_coeff - 2 * B_coeff - 3 * B_coeff**2
    c0 = -(A_coeff * B_coeff - B_coeff**2 - B_coeff**3)

    roots_z = np.roots([1.0, c2, c1, c0])
    
    # Extract real roots (accounting for minor floating point inaccuracies in imaginary parts)
    real_roots = [z.real for z in roots_z if abs(z.imag) < 1e-9 and z.real > 0]

    if not real_roots:
        raise ValueError(
            f"No valid positive real roots found for compressibility factor (Z) "
            f"at P={pressure:.2e} Pa and T={temperature:.2f} K."
        )

    # Return the maximum root (gas/vapor phase)
    return max(real_roots)