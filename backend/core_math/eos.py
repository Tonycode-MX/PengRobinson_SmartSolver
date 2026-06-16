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


def calculate_advanced_pr_parameters(
    temperature: float, tc: float, pc: float, omega: float
) -> Tuple[float, float, float]:
    """
    Calculates the Peng-Robinson 'a' and 'b' parameters, along with the 
    analytical temperature derivative da/dT.

    Args:
        temperature (float): Absolute temperature in Kelvin (K).
        tc (float): Critical temperature in Kelvin (K).
        pc (float): Critical pressure in Pascals (Pa).
        omega (float): Acentric factor (dimensionless).

    Returns:
        Tuple[float, float, float]: Parameter 'a', parameter 'b', and derivative da/dT.
    """
    if temperature <= 0 or tc <= 0 or pc <= 0:
        raise ValueError("Temperature, critical temperature, and critical pressure must be strictly positive.")

    tr = temperature / tc
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + kappa * (1 - np.sqrt(tr))) ** 2

    a = 0.45724 * (UNIVERSAL_GAS_CONSTANT**2 * tc**2) / pc * alpha
    b = 0.07780 * (UNIVERSAL_GAS_CONSTANT * tc) / pc

    # Analytical derivative of alpha with respect to T: d(alpha)/dT
    dalpha_dT = -kappa * (1 + kappa * (1 - np.sqrt(tr))) / (tc * np.sqrt(tr))
    # Analytical derivative of 'a' with respect to T: da/dT
    da_dT = 0.45724 * (UNIVERSAL_GAS_CONSTANT**2 * tc**2) / pc * dalpha_dT

    return a, b, da_dT


def calculate_pressure(
    volume: float, temperature: float, tc: float, pc: float, omega: float
) -> float:
    # ... (docstring y validaciones iguales) ...
    if volume <= 0:
        raise ValueError("Molar volume must be strictly positive (V > 0).")
    if temperature <= 0:
        raise ValueError("Temperature must be strictly positive (T > 0).")

    # Corrección: desempaquetar correctamente los 3 valores, ignorando da_dT con '_'
    a, b, _ = calculate_advanced_pr_parameters(temperature, tc, pc, omega)
    
    # Peng-Robinson Equation
    term1 = (UNIVERSAL_GAS_CONSTANT * temperature) / (volume - b)
    term2 = a / (volume * (volume + b) + b * (volume - b))
    
    return term1 - term2


def calculate_volume(
    pressure: float, temperature: float, tc: float, pc: float, omega: float
) -> float:
    # ... (docstring y validaciones iguales) ...
    if pressure <= 0:
        raise ValueError("Pressure must be strictly positive (P > 0).")
    if temperature <= 0:
        raise ValueError("Temperature must be strictly positive (T > 0).")

    # Corrección: desempaquetar correctamente
    a, b, _ = calculate_advanced_pr_parameters(temperature, tc, pc, omega)
    
    # Reutilizar la función de Z en lugar de repetir la lógica cúbica
    z_max = calculate_compressibility_factor(pressure, temperature, a, b)
    
    # Despejar el volumen
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


def calculate_compressibility_factor(
    pressure: float, temperature: float, a: float, b: float
) -> float:
    """
    Solves the cubic Peng-Robinson polynomial to find the compressibility factor (Z).
    Selects the maximum real root corresponding to the stable vapor/gas phase.
    """
    A_coeff = (a * pressure) / (UNIVERSAL_GAS_CONSTANT**2 * temperature**2)
    B_coeff = (b * pressure) / (UNIVERSAL_GAS_CONSTANT * temperature)

    # Cubic equation coefficients: Z^3 + c2*Z^2 + c1*Z + c0 = 0
    c2 = -(1 - B_coeff)
    c1 = A_coeff - 2 * B_coeff - 3 * B_coeff**2
    c0 = -(A_coeff * B_coeff - B_coeff**2 - B_coeff**3)

    roots_z = np.roots([1, c2, c1, c0])
    real_roots = [z.real for z in roots_z if abs(z.imag) < 1e-9 and z.real > 0]

    if not real_roots:
        raise ValueError("No valid positive real roots found for compressibility factor (Z).")

    return max(real_roots)
