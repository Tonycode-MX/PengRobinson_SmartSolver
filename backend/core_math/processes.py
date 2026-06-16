import numpy as np
import CoolProp.CoolProp as CP
from typing import Tuple, Dict, List

# Universal gas constant in J/(mol*K)
UNIVERSAL_GAS_CONSTANT = 8.314462618

def calculate_residual_properties(
    pressure: float, temperature: float, tc: float, pc: float, omega: float
) -> Dict[str, float]:
    """
    Calculates advanced residual thermodynamic properties using the Peng-Robinson EOS.

    Args:
        pressure (float): Pressure in Pascals (Pa).
        temperature (float): Absolute temperature in Kelvin (K).
        tc (float): Critical temperature (K).
        pc (float): Critical pressure (Pa).
        omega (float): Acentric factor.

    Returns:
        Dict[str, float]: Dictionary containing Z, Fugacity coeff, Residual H, Residual S, and V.
    """
    if pressure <= 0 or temperature <= 0:
        raise ValueError("Pressure and temperature must be strictly positive.")

    # 1. Compute fundamental and derivative EOS parameters
    a, b, da_dT = calculate_advanced_pr_parameters(temperature, tc, pc, omega)
    z = calculate_compressibility_factor(pressure, temperature, a, b)

    # Dimensionless EOS parameters
    A = (a * pressure) / (UNIVERSAL_GAS_CONSTANT**2 * temperature**2)
    B = (b * pressure) / (UNIVERSAL_GAS_CONSTANT * temperature)

    # 2. Fugacity Coefficient (phi) Calculation
    sqrt_2 = np.sqrt(2)
    term_fug_1 = z - 1 - np.log(z - B)
    term_fug_2 = (A / (2 * sqrt_2 * B)) * np.log((z + (1 + sqrt_2) * B) / (z + (1 - sqrt_2) * B))
    ln_phi = term_fug_1 - term_fug_2
    phi = np.exp(ln_phi)

    # 3. Residual Enthalpy (H^R) Calculation in J/mol
    term_h_1 = UNIVERSAL_GAS_CONSTANT * temperature * (z - 1)
    term_h_2 = (temperature * da_dT - a) / (2 * sqrt_2 * b)
    term_h_3 = np.log((z + (1 + sqrt_2) * B) / (z + (1 - sqrt_2) * B))
    h_residual = term_h_1 + term_h_2 * term_h_3

    # 4. Residual Entropy (S^R) Calculation in J/(mol*K)
    term_s_1 = UNIVERSAL_GAS_CONSTANT * np.log(z - B)
    term_s_2 = da_dT / (2 * sqrt_2 * b)
    term_s_3 = np.log((z + (1 + sqrt_2) * B) / (z + (1 - sqrt_2) * B))
    s_residual = term_s_1 + term_s_2 * term_s_3

    # Derived molar volume for subsequent thermal calculations
    v_molar = z * UNIVERSAL_GAS_CONSTANT * temperature / pressure

    return {
        "compressibility_factor_Z": z,
        "fugacity_coefficient_phi": phi,
        "residual_enthalpy_Hr": h_residual,
        "residual_entropy_Sr": s_residual,
        "molar_volume_V": v_molar,
    }


def calculate_joule_thomson(
    pressure: float, temperature: float, tc: float, pc: float, omega: float, cp_ideal: float
) -> float:
    """
    Calculates the Joule-Thomson Coefficient (mu_JT) in K/Pa.
    Requires the ideal gas Cp of the substance at the given temperature.
    """
    # Numerical differentiation to find (dV/dT)_P
    dT = 1e-3
    props_low = calculate_residual_properties(pressure, temperature - dT, tc, pc, omega)
    props_high = calculate_residual_properties(pressure, temperature + dT, tc, pc, omega)
    
    v_low = props_low["molar_volume_V"]
    v_high = props_high["molar_volume_V"]
    
    # Partial derivative (dV/dT) at constant Pressure
    dv_dT_p = (v_high - v_low) / (2 * dT)

    # Extract central state properties
    props_central = calculate_residual_properties(pressure, temperature, tc, pc, omega)
    v_molar = props_central["molar_volume_V"]
    
    # Correct ideal Cp to real Cp: Cp_real = Cp_ideal + d(H^R)/dT
    h_low = props_low["residual_enthalpy_Hr"]
    h_high = props_high["residual_enthalpy_Hr"]
    dHr_dT_p = (h_high - h_low) / (2 * dT)
    cp_real = cp_ideal + dHr_dT_p

    # Classical Joule-Thomson relation: mu_JT = (1 / Cp_real) * [ T * (dV/dT)_P - V ]
    mu_jt = (1.0 / cp_real) * (temperature * dv_dT_p - v_molar)
    return mu_jt

def get_mixture_pure_properties(fluids: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts critical properties (Tc, Pc, omega) for all pure components 
    present in the chemical mixture using CoolProp.

    Args:
        fluids (List[str]): List of fluid names (e.g., ['Methane', 'Ethane', 'CO2']).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of Tc (K), Pc (Pa), and omega.
    """
    tc_list, pc_list, omega_list = [], [], []
    for fluid in fluids:
        try:
            tc_list.append(CP.PropsSI("Tcrit", fluid))
            pc_list.append(CP.PropsSI("pcrit", fluid))
            omega_list.append(CP.PropsSI("acentric", fluid))
        except ValueError as e:
            raise ValueError(f"Fluid '{fluid}' is not recognized in CoolProp database. Error: {e}")
            
    return np.array(tc_list), np.array(pc_list), np.array(omega_list)


def calculate_pure_pr_parameters(
    temperature: float, tc: np.ndarray, pc: np.ndarray, omega: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the individual 'a', 'b' and analytical temperature derivative (da/dT)
    vectors for all components at the system temperature.
    """
    if temperature <= 0:
        raise ValueError("System temperature must be strictly positive (T > 0).")

    tr = temperature / tc
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + kappa * (1 - np.sqrt(tr))) ** 2

    a_pure = 0.45724 * (UNIVERSAL_GAS_CONSTANT**2 * tc**2) / pc * alpha
    b_pure = 0.07780 * (UNIVERSAL_GAS_CONSTANT * tc) / pc

    # Analytical temperature derivative of alpha parameter: d(alpha)/dT
    dalpha_dT = -kappa * (1 + kappa * (1 - np.sqrt(tr))) / (tc * np.sqrt(tr))
    # Analytical temperature derivative of attraction parameter: da/dT
    da_dT_pure = 0.45724 * (UNIVERSAL_GAS_CONSTANT**2 * tc**2) / pc * dalpha_dT

    return a_pure, b_pure, da_dT_pure


def apply_van_der_waals_mixing_rules(
    mole_fractions: np.ndarray, a_pure: np.ndarray, b_pure: np.ndarray, 
    da_dT_pure: np.ndarray, kij_matrix: np.ndarray
) -> Tuple[float, float, float]:
    """
    Applies classical van der Waals mixing rules to evaluate the mixture parameters 
    am (attraction), bm (covolume), and the overall temperature derivative dam/dT.
    """
    num_components = len(mole_fractions)
    
    # 1. Mixture covolume (bm) - Linear scaling rule
    bm = float(np.sum(mole_fractions * b_pure))
    
    # 2. Mixture attraction (am) and derivative (dam_dT) via cross-interactions matrix
    am = 0.0
    dam_dT = 0.0
    
    for i in range(num_components):
        for j in range(num_components):
            # Cross-attraction parameter (Geometric mean rule corrected by kij)
            a_ij = np.sqrt(a_pure[i] * a_pure[j]) * (1.0 - kij_matrix[i, j])
            am += mole_fractions[i] * mole_fractions[j] * a_ij
            
            # Analytical derivative of cross-attraction: da_ij/dT
            # d(sqrt(a_i*a_j))/dT = 0.5 * (a_j*da_i/dT + a_i*da_j/dT) / sqrt(a_i*a_j)
            da_ij_dT = 0.5 * (a_pure[j] * da_dT_pure[i] + a_pure[i] * da_dT_pure[j]) / np.sqrt(a_pure[i] * a_pure[j])
            da_ij_dT_corrected = da_ij_dT * (1.0 - kij_matrix[i, j])
            dam_dT += mole_fractions[i] * mole_fractions[j] * da_ij_dT_corrected

    return am, bm, dam_dT


def calculate_mixture_z_factor(pressure: float, temperature: float, am: float, bm: float) -> float:
    """
    Solves the cubic Peng-Robinson EOS polynomial using unified mixture parameters 
    to extract the real compressibility factor (Z).
    """
    A = (am * pressure) / (UNIVERSAL_GAS_CONSTANT**2 * temperature**2)
    B = (bm * pressure) / (UNIVERSAL_GAS_CONSTANT * temperature)

    # Polynomial coefficients layout: Z^3 + c2*Z^2 + c1*Z + c0 = 0
    c2 = -(1.0 - B)
    c1 = A - 2.0 * B - 3.0 * B**2
    c0 = -(A * B - B**2 - B**3)

    roots_z = np.roots([1.0, c2, c1, c0])
    real_roots = [z.real for z in roots_z if abs(z.imag) < 1e-9 and z.real > 0]

    if not real_roots:
        raise ValueError("Thermodynamic state error: No valid positive roots found for mixture Z.")

    return max(real_roots)


def calculate_mixture_properties(
    pressure: float, temperature: float, fluids: List[str], mole_fractions: List[float], kij_matrix: np.ndarray
) -> Dict[str, float]:
    """
    Master function to solve the thermodynamic equation of state for a multicomponent gas.

    Returns a comprehensive dictionary with unified system attributes.
    """
    x = np.array(mole_fractions)
    if not np.isclose(np.sum(x), 1.0):
        raise ValueError("Composition definition error: Mole fractions must sum up to exactly 1.0.")
    if pressure <= 0:
        raise ValueError("System pressure must be strictly positive (P > 0).")

    # 1. Fetch pure data and solve local parameters
    tc, pc, omega = get_mixture_pure_properties(fluids)
    a_p, b_p, da_dT_p = calculate_pure_pr_parameters(temperature, tc, pc, omega)
    
    # 2. Merge properties using mixing rules
    am, bm, dam_dT = apply_van_der_waals_mixing_rules(x, a_p, b_p, da_dT_p, kij_matrix)
    
    # 3. Solve master mixture Z-factor
    z_mixture = calculate_mixture_z_factor(pressure, temperature, am, bm)

    # 4. Dimensionless system parameters
    A = (am * pressure) / (UNIVERSAL_GAS_CONSTANT**2 * temperature**2)
    B = (bm * pressure) / (UNIVERSAL_GAS_CONSTANT * temperature)
    
    # 5. Extract caloric residual attributes for the mixture
    sqrt_2 = np.sqrt(2)
    log_term = np.log((z_mixture + (1 + sqrt_2) * B) / (z_mixture + (1 - sqrt_2) * B))
    
    # Combined Residual Enthalpy (Hm^R)
    term_h1 = UNIVERSAL_GAS_CONSTANT * temperature * (z_mixture - 1.0)
    term_h2 = (temperature * dam_dT - am) / (2.0 * sqrt_2 * bm)
    h_res_mix = term_h1 + term_h2 * log_term
    
    # Combined Residual Entropy (Sm^R)
    term_s1 = UNIVERSAL_GAS_CONSTANT * np.log(z_mixture - B)
    term_s2 = dam_dT / (2.0 * sqrt_2 * bm)
    s_res_mix = term_s1 + term_s2 * log_term

    # Unified real molar volume
    v_molar_mix = z_mixture * UNIVERSAL_GAS_CONSTANT * temperature / pressure

    return {
        "mixture_Z": z_mixture,
        "mixture_Hr_J_mol": h_res_mix,
        "mixture_Sr_J_molK": s_res_mix,
        "mixture_V_m3_mol": v_molar_mix
    }