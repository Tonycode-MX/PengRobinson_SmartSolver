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
import numpy as np
import CoolProp.CoolProp as CP
from typing import Tuple, Dict

# Universal gas constant in J/(mol*K)
UNIVERSAL_GAS_CONSTANT = 8.314462618


def get_gas_properties(fluid: str) -> Tuple[float, float, float]:
    """
    Extracts the critical properties of a given fluid using CoolProp.

    Args:
        fluid (str): The name of the fluid (e.g., 'Methane', 'Nitrogen').

    Returns:
        Tuple[float, float, float]: Critical temperature (Tc) in K, 
                                     Critical pressure (Pc) in Pa, 
                                     Acentric factor (omega), dimensionless.
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


# --- Testing and Demonstration Block ---
if __name__ == "__main__":
    print("=========================================================")
    print("   ADVANCED RESIDUAL PROPERTIES ENGINE (PR EOS)          ")
    print("=========================================================\n")

    fluid_name = "Methane"
    
    try:
        # 1. Load Critical Constants
        t_crit, p_crit, acentric = get_gas_properties(fluid_name)
        print(f"[{fluid_name}] Critical Constants Loaded:")
        print(f"Tc = {t_crit:.2f} K | Pc = {p_crit:.2f} Pa | Omega = {acentric:.4f}\n")

        # Test State Conditions (High pressure gas)
        test_T = 280.0       # Kelvin
        test_P = 5.0e6       # 50.0 bar (5,000,000 Pa)

        # 2. Compute State and Residual Properties
        results = calculate_residual_properties(test_P, test_T, t_crit, p_crit, acentric)

        print(f"--- Thermodynamic Results at {test_T} K and {test_P/1e5:.1f} bar ---")
        print(f"Compressibility Factor (Z)   : {results['compressibility_factor_Z']:.5f}")
        print(f"Fugacity Coefficient (phi)  : {results['fugacity_coefficient_phi']:.5f}")
        print(f"Residual Enthalpy (H^R)      : {results['residual_enthalpy_Hr']:.2f} J/mol")
        print(f"Residual Entropy (S^R)       : {results['residual_entropy_Sr']:.2f} J/(mol*K)")
        print(f"Real Molar Volume (V)        : {results['molar_volume_V']:.7f} m^3/mol")

        # 3. Fetch Ideal Gas Cp at 1 atm reference to avoid numerical singularities
        cp_ideal_molar = CP.PropsSI("Cpmolar", "T", test_T, "P", 101325, fluid_name)
        
        # 4. Compute Joule-Thomson Coefficient
        mu_jt = calculate_joule_thomson(test_P, test_T, t_crit, p_crit, acentric, cp_ideal_molar)
        
        print(f"\n--- Thermal Response Coeff ---")
        print(f"Joule-Thomson Coeff (mu_JT) : {mu_jt * 1e6:.4f} K/MPa")
        if mu_jt > 0:
            print("-> Effect: The gas WILL COOL DOWN upon isoenthalpic expansion (throttling valve).")
        else:
            print("-> Effect: The gas WILL HEAT UP upon isoenthalpic expansion (throttling valve).")

        # 5. Robustness Validation Test
        print("\n--- System Robustness Check ---")
        try:
            calculate_residual_properties(-5000, test_T, t_crit, p_crit, acentric)
        except ValueError as e:
            print(f"Boundary protection active: {e}")

    except Exception as e:
        print(f"An unexpected error occurred in the system: {e}")
        import numpy as np
import CoolProp.CoolProp as CP
from typing import List, Tuple, Dict

# Universal gas constant in J/(mol*K)
UNIVERSAL_GAS_CONSTANT = 8.314462618


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


# --- Execution and Demonstration Entry Point ---
if __name__ == "__main__":
    print("=========================================================")
    print("   MULTICOMPONENT MIXTURE EOS SOLVER (PENG-ROBINSON)    ")
    print("=========================================================\n")

    # Target Mixture Definition: Natural Gas Surrogate (3 components)
    gas_components = ["Methane", "Ethane", "Propane"]
    fractions = [0.85, 0.10, 0.05]  # 85% CH4, 10% C2H6, 5% C3H8
    
    # Symmetric Binary Interaction Parameter Matrix (kij)
    # Row/Col order follows gas_components layout
    kij = np.array([
        [0.000, 0.002, 0.005],  # CH4 interactions
        [0.002, 0.000, 0.001],  # C2H6 interactions
        [0.005, 0.001, 0.000]   # C3H8 interactions
    ])

    # Simulation conditions (High-pressure pipeline simulation state)
    system_T = 290.0      # Kelvin
    system_P = 6.5e6      # 65.0 bar (6,500,000 Pa)

    try:
        results = calculate_mixture_properties(system_P, system_T, gas_components, fractions, kij)
        
        print("--- Gas Composition Stream ---")
        for comp, frac in zip(gas_components, fractions):
            print(f" * {comp:<8}: {frac*100:>5.1f}%")
            
        print(f"\n--- System State Conditions: {system_T} K @ {system_P/1e5:.1f} bar ---")
        print(f"Mixture Compressibility Factor (Z) : {results['mixture_Z']:.5f}")
        print(f"Mixture Molar Volume (V_m)         : {results['mixture_V_m3_mol']:.7f} m^3/mol")
        print(f"Mixture Residual Enthalpy (H^R)    : {results['mixture_Hr_J_mol']:.2f} J/mol")
        print(f"Mixture Residual Entropy (S^R)     : {results['mixture_Sr_J_molK']:.2f} J/(mol*K)")

        # Boundary Protection Check
        print("\n--- System Boundary Check ---")
        calculate_mixture_properties(system_P, system_T, gas_components, [0.5, 0.5, 0.1], kij) # Bad fractions

    except ValueError as e:
        print(f"Boundary protection active: {e}")
    except Exception as e:
        print(f"An unexpected system failure occurred: {e}")