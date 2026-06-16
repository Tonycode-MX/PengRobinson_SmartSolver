import numpy as np
import scipy.optimize as opt
import CoolProp.CoolProp as CP
from typing import Tuple

# Universal gas constant in J/(mol*K)
UNIVERSAL_GAS_CONSTANT = 8.31446


def get_gas_properties(fluid: str) -> Tuple[float, float, float]:
    """
    Extracts the critical properties of a given fluid using CoolProp.

    Args:
        fluid (str): The name of the fluid (e.g., 'Methane', 'Water').

    Returns:
        Tuple[float, float, float]: A tuple containing:
            - Critical temperature (Tc) in Kelvin (K).
            - Critical pressure (Pc) in Pascals (Pa).
            - Acentric factor (omega), dimensionless.

    Raises:
        ValueError: If the fluid is not available or recognized in CoolProp.
    """
    try:
        tc = float(CP.PropsSI("Tcrit", fluid))
        pc = float(CP.PropsSI("pcrit", fluid))
        omega = float(CP.PropsSI("acentric", fluid))
        return tc, pc, omega
    except ValueError as e:
        raise ValueError(f"Fluid '{fluid}' is not available in CoolProp. Original error: {e}")


def calculate_pr_parameters(
    temperature: float, tc: float, pc: float, omega: float
) -> Tuple[float, float]:
    """
    Calculates the 'a' and 'b' parameters for the Peng-Robinson equation of state.

    Args:
        temperature (float): Absolute temperature in Kelvin (K).
        tc (float): Critical temperature in Kelvin (K).
        pc (float): Critical pressure in Pascals (Pa).
        omega (float): Acentric factor (dimensionless).

    Returns:
        Tuple[float, float]: The calculated 'a' and 'b' parameters.

    Raises:
        ValueError: If temperature, tc, or pc are less than or equal to zero.
    """
    if temperature <= 0 or tc <= 0 or pc <= 0:
        raise ValueError("Temperature, critical temperature, and critical pressure must be strictly positive.")

    tr = temperature / tc
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + kappa * (1 - np.sqrt(tr))) ** 2

    a = 0.45724 * (UNIVERSAL_GAS_CONSTANT**2 * tc**2) / pc * alpha
    b = 0.07780 * (UNIVERSAL_GAS_CONSTANT * tc) / pc
    
    return a, b


def calculate_pressure(
    volume: float, temperature: float, tc: float, pc: float, omega: float
) -> float:
    """
    Calculates the pressure of a gas using the Peng-Robinson equation of state.

    Args:
        volume (float): Molar volume in m^3/mol.
        temperature (float): Absolute temperature in Kelvin (K).
        tc (float): Critical temperature in Kelvin (K).
        pc (float): Critical pressure in Pascals (Pa).
        omega (float): Acentric factor (dimensionless).

    Returns:
        float: Calculated pressure in Pascals (Pa).

    Raises:
        ValueError: If volume or temperature are less than or equal to zero.
    """
    if volume <= 0:
        raise ValueError("Molar volume must be strictly positive (V > 0).")
    if temperature <= 0:
        raise ValueError("Temperature must be strictly positive (T > 0).")

    a, b = calculate_pr_parameters(temperature, tc, pc, omega)
    
    # Peng-Robinson Equation
    term1 = (UNIVERSAL_GAS_CONSTANT * temperature) / (volume - b)
    term2 = a / (volume * (volume + b) + b * (volume - b))
    
    return term1 - term2


def calculate_volume(
    pressure: float, temperature: float, tc: float, pc: float, omega: float
) -> float:
    """
    Calculates the molar volume of a gas given pressure and temperature.
    
    Solves for the compressibility factor (Z) using the cubic Peng-Robinson form.

    Args:
        pressure (float): Pressure in Pascals (Pa).
        temperature (float): Absolute temperature in Kelvin (K).
        tc (float): Critical temperature in Kelvin (K).
        pc (float): Critical pressure in Pascals (Pa).
        omega (float): Acentric factor (dimensionless).

    Returns:
        float: Calculated molar volume in m^3/mol (maximum real root).

    Raises:
        ValueError: If pressure or temperature are less than or equal to zero, 
                    or if no valid positive real roots are found.
    """
    if pressure <= 0:
        raise ValueError("Pressure must be strictly positive (P > 0).")
    if temperature <= 0:
        raise ValueError("Temperature must be strictly positive (T > 0).")

    a, b = calculate_pr_parameters(temperature, tc, pc, omega)
    
    A_coeff = (a * pressure) / (UNIVERSAL_GAS_CONSTANT**2 * temperature**2)
    B_coeff = (b * pressure) / (UNIVERSAL_GAS_CONSTANT * temperature)
    
    # Coefficients for the cubic equation: Z^3 + c2*Z^2 + c1*Z + c0 = 0
    c2 = -(1 - B_coeff)
    c1 = A_coeff - 2 * B_coeff - 3 * B_coeff**2
    c0 = -(A_coeff * B_coeff - B_coeff**2 - B_coeff**3)
    
    roots_z = np.roots([1, c2, c1, c0])
    
    # Extract real roots (accounting for minor floating point inaccuracies in imaginary parts)
    real_roots = [z.real for z in roots_z if abs(z.imag) < 1e-9 and z.real > 0]
    
    if not real_roots:
        raise ValueError("No valid positive real roots found for compressibility factor (Z).")
        
    z_max = max(real_roots)
    return z_max * UNIVERSAL_GAS_CONSTANT * temperature / pressure


def calculate_temperature(
    pressure: float, volume: float, tc: float, pc: float, omega: float
) -> float:
    """
    Calculates the temperature of a gas given its pressure and molar volume.

    Uses Brent's method to find the root of the Peng-Robinson pressure function.

    Args:
        pressure (float): Pressure in Pascals (Pa).
        volume (float): Molar volume in m^3/mol.
        tc (float): Critical temperature in Kelvin (K).
        pc (float): Critical pressure in Pascals (Pa).
        omega (float): Acentric factor (dimensionless).

    Returns:
        float: Calculated absolute temperature in Kelvin (K).

    Raises:
        ValueError: If pressure or volume are less than or equal to zero, 
                    or if the root-finding algorithm fails to converge.
    """
    if pressure <= 0:
        raise ValueError("Pressure must be strictly positive (P > 0).")
    if volume <= 0:
        raise ValueError("Molar volume must be strictly positive (V > 0).")

    def objective_function(t_est: float) -> float:
        return calculate_pressure(volume, t_est, tc, pc, omega) - pressure

    try:
        solution = opt.root_scalar(
            objective_function, bracket=[tc * 0.1, tc * 5], method="brentq"
        )
        if not solution.converged:
            raise ValueError("Root-finding algorithm did not converge.")
        return float(solution.root)
    except ValueError as e:
        raise ValueError(f"Failed to find a valid temperature: {e}")


'''--- Testing / Examples ---
This section includes example usage and error validation tests for the implemented functions.

if __name__ == "__main__":
    # --- Testing / Examples ---
    print("--- Peng-Robinson EOS Calculator Tests ---")
    
    fluid_name = "Methane"
    try:
        # 1. Get properties
        t_crit, p_crit, acentric = get_gas_properties(fluid_name)
        print(f"[{fluid_name}] Critical Properties Extracted:")
        print(f"Tc = {t_crit:.2f} K, Pc = {p_crit:.2f} Pa, Omega = {acentric:.4f}\n")

        # Define test state
        test_T = 300.0  # Kelvin
        test_P = 101325.0  # Pascals (1 atm)
        
        # 2. Calculate Volume from T and P
        calc_V = calculate_volume(test_P, test_T, t_crit, p_crit, acentric)
        print(f"Calculated Volume (at {test_T} K, {test_P} Pa): {calc_V:.6f} m^3/mol")

        # 3. Calculate Pressure from V and T (should match test_P)
        calc_P = calculate_pressure(calc_V, test_T, t_crit, p_crit, acentric)
        print(f"Calculated Pressure (from {calc_V:.6f} m^3/mol, {test_T} K): {calc_P:.2f} Pa")

        # 4. Calculate Temperature from P and V (should match test_T)
        calc_T = calculate_temperature(test_P, calc_V, t_crit, p_crit, acentric)
        print(f"Calculated Temperature (from {calc_V:.6f} m^3/mol, {test_P} Pa): {calc_T:.2f} K\n")

        # 5. Error Validation Testing
        print("--- Testing Error Validation ---")
        try:
            calculate_volume(-100, test_T, t_crit, p_crit, acentric)
        except ValueError as e:
            print(f"Successfully caught error for negative pressure: {e}")

        try:
            calculate_pressure(calc_V, -50, t_crit, p_crit, acentric)
        except ValueError as e:
            print(f"Successfully caught error for negative temperature: {e}")

    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")

'''
