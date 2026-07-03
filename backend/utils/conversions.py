
def convert_temperature_to_si(value: float, from_unit: str) -> float:
    """
    Converts temperature to the base SI unit Kelvin (K).
    
    Expected from_unit inputs: 'C', 'Celsius', 'F', 'Fahrenheit', 'R', 'Rankine', 'K', 'Kelvin'.
    The conversion is case-insensitive and ignores trailing/leading whitespaces.
    
    Args:
        value (float): The temperature value to convert.
        from_unit (str): The original unit of the temperature.
        
    Returns:
        float: Absolute temperature in Kelvin (K).
        
    Raises:
        ValueError: If the unit is not recognized or supported.
        ValueError: If the resulting temperature is below absolute zero (< 0 K).
    """
    # Sanitize the input to handle messy agent strings
    unit = from_unit.strip().lower()
    
    if unit in ['k', 'kelvin']:
        kelvin_value = value
    elif unit in ['c', 'celsius']:
        kelvin_value = value + 273.15
    elif unit in ['f', 'fahrenheit']:
        kelvin_value = (value - 32.0) * 5.0 / 9.0 + 273.15
    elif unit in ['r', 'rankine']:
        kelvin_value = value * 5.0 / 9.0
    else:
        raise ValueError(
            f"Unrecognized temperature unit: '{from_unit}'. "
            "Supported units are: K, Celsius, F, Rankine."
        )
        
    # Thermodynamic reality check
    if kelvin_value < 0:
        raise ValueError(
            f"Physically impossible state: Calculated temperature ({kelvin_value:.2f} K) "
            "is below absolute zero."
        )
        
    return kelvin_value

def convert_pressure_to_si(value: float, from_unit: str) -> float:
    """
    Converts pressure to the base SI unit Pascals (Pa). 
    Assumes the input represents absolute pressure, which is required for 
    Equations of State (EOS) calculations.
    
    Expected from_unit inputs: 'atm', 'bar', 'psi', 'kPa', 'MPa', 'mmHg', 'Torr', 'Pa'.
    The conversion is case-insensitive and ignores trailing/leading whitespaces.
    
    Args:
        value (float): The absolute pressure value to convert.
        from_unit (str): The original unit of the pressure.
        
    Returns:
        float: Absolute pressure in Pascals (Pa).
        
    Raises:
        ValueError: If the unit is not recognized or supported.
        ValueError: If the resulting absolute pressure is negative.
    """
    # Sanitize the input to handle messy agent strings
    unit = from_unit.strip().lower()
    
    if unit in ['pa', 'pascal', 'pascals']:
        pa_value = value
    elif unit in ['kpa', 'kilopascal', 'kilopascals']:
        pa_value = value * 1000.0
    elif unit in ['mpa', 'megapascal', 'megapascals']:
        pa_value = value * 1000000.0
    elif unit in ['atm', 'atmosphere', 'atmospheres']:
        pa_value = value * 101325.0
    elif unit in ['bar', 'bars']:
        pa_value = value * 100000.0
    elif unit in ['psi', 'lbf/in2']:
        pa_value = value * 6894.75729
    elif unit in ['mmhg']:
        pa_value = value * 133.322387415
    elif unit in ['torr']:
        pa_value = value * (101325.0 / 760.0) # Exact Torr definition
    else:
        raise ValueError(
            f"Unrecognized pressure unit: '{from_unit}'. "
            "Supported units are: Pa, kPa, MPa, atm, bar, psi, mmHg, Torr."
        )
        
    # Thermodynamic reality check for absolute pressure
    if pa_value < 0:
        raise ValueError(
            f"Physically impossible state: Calculated absolute pressure ({pa_value:.2f} Pa) "
            "cannot be negative. EOS models require absolute pressure."
        )
        
    return pa_value


def convert_volume_to_si(value: float, from_unit: str) -> float:
    """
    Converts total volume to the base SI unit cubic meters (m^3).
    
    Expected from_unit inputs: 'L', 'liter', 'mL', 'cm3', 'cc', 'ft3', 'gal', 'm3'.
    The conversion is case-insensitive and ignores trailing/leading whitespaces.
    
    Args:
        value (float): The total volume value to convert.
        from_unit (str): The original unit of the volume.
        
    Returns:
        float: Total volume in cubic meters (m^3).
        
    Raises:
        ValueError: If the unit is not recognized or supported.
        ValueError: If the resulting volume is less than or equal to zero.
    """
    # Sanitize the input to handle messy agent strings
    unit = from_unit.strip().lower()
    
    if unit in ['m3', 'm^3', 'cubic meter', 'cubic meters']:
        m3_value = value
    elif unit in ['l', 'liter', 'liters', 'litre', 'litres']:
        m3_value = value * 1e-3
    elif unit in ['ml', 'milliliter', 'milliliters', 'cm3', 'cm^3', 'cc', 'cubic centimeter']:
        m3_value = value * 1e-6
    elif unit in ['ft3', 'ft^3', 'cubic foot', 'cubic feet']:
        m3_value = value * 0.028316846592
    elif unit in ['gal', 'gallon', 'gallons', 'us gal']:
        m3_value = value * 0.00378541
    else:
        raise ValueError(
            f"Unrecognized volume unit: '{from_unit}'. "
            "Supported units are: m3, L, mL, cm3, cc, ft3, gal."
        )
        
    # Thermodynamic reality check for volume
    if m3_value <= 0:
        raise ValueError(
            f"Physically impossible state: Calculated volume ({m3_value:.5e} m^3) "
            "must be strictly positive (V > 0)."
        )
        
    return m3_value


def convert_molar_volume_to_si(value: float, from_unit: str) -> float:
    """
    Converts molar volume to the base SI unit cubic meters per mole (m^3/mol).
    
    Expected from_unit inputs: 'L/mol', 'mL/mol', 'cm3/mol', 'ft3/lbmol', 'm3/kmol', 'm3/mol'.
    The conversion is case-insensitive and ignores trailing/leading whitespaces.
    
    Args:
        value (float): The molar volume value to convert.
        from_unit (str): The original unit of the molar volume.
        
    Returns:
        float: Molar volume in cubic meters per mole (m^3/mol).
        
    Raises:
        ValueError: If the unit is not recognized or supported.
        ValueError: If the resulting molar volume is less than or equal to zero.
    """
    # Sanitize the input to handle messy agent strings
    unit = from_unit.strip().lower()
    
    if unit in ['m3/mol', 'm^3/mol']:
        m3_mol_value = value
    elif unit in ['l/mol', 'liter/mol', 'liters/mol', 'm3/kmol', 'm^3/kmol']:
        # 1 L = 1e-3 m^3, and 1 m^3/kmol = 1/1000 m^3/mol
        m3_mol_value = value * 1e-3
    elif unit in ['ml/mol', 'cm3/mol', 'cm^3/mol', 'cc/mol']:
        # 1 cm^3 = 1e-6 m^3
        m3_mol_value = value * 1e-6
    elif unit in ['ft3/lbmol', 'ft^3/lbmol']:
        # 1 ft^3 = 0.028316846592 m^3
        # 1 lb-mol = 453.59237 mol
        m3_mol_value = value * (0.028316846592 / 453.59237)
    else:
        raise ValueError(
            f"Unrecognized molar volume unit: '{from_unit}'. "
            "Supported units are: m3/mol, L/mol, cm3/mol, m3/kmol, ft3/lbmol."
        )
        
    # Thermodynamic reality check for molar volume
    if m3_mol_value <= 0:
        raise ValueError(
            f"Physically impossible state: Calculated molar volume ({m3_mol_value:.5e} m^3/mol) "
            "must be strictly positive (v > 0)."
        )
        
    return m3_mol_value


def convert_energy_to_si(value: float, from_unit: str) -> float:
    """
    Converts total energy (Work, Heat, etc.) to the base SI unit Joules (J).
    
    Expected from_unit inputs: 'J', 'kJ', 'cal', 'kcal', 'BTU', 'erg', 'eV'.
    The conversion is case-insensitive and ignores trailing/leading whitespaces.
    
    Note: Unlike absolute pressure or volume, total energy can mathematically and 
    physically be negative in thermodynamics (e.g., heat leaving a system, 
    expansion work, or negative residual properties). Therefore, there is no 
    strictly positive validation check applied to the input value.
    
    Args:
        value (float): The energy value to convert.
        from_unit (str): The original unit of the energy.
        
    Returns:
        float: Energy in Joules (J).
        
    Raises:
        ValueError: If the unit is not recognized or supported.
    """
    # Sanitize the input to handle messy agent strings
    unit = from_unit.strip().lower()
    
    if unit in ['j', 'joule', 'joules']:
        j_value = value
    elif unit in ['kj', 'kilojoule', 'kilojoules']:
        j_value = value * 1000.0
    elif unit in ['cal', 'calorie', 'calories']:
        # Thermochemical calorie
        j_value = value * 4.184
    elif unit in ['kcal', 'kilocalorie', 'kilocalories']:
        j_value = value * 4184.0
    elif unit in ['btu', 'british thermal unit']:
        # International Table BTU
        j_value = value * 1055.056
    elif unit in ['erg', 'ergs']:
        j_value = value * 1e-7
    elif unit in ['ev', 'electronvolt', 'electronvolts']:
        j_value = value * 1.602176634e-19
    else:
        raise ValueError(
            f"Unrecognized energy unit: '{from_unit}'. "
            "Supported units are: J, kJ, cal, kcal, BTU, erg, eV."
        )
        
    return j_value


def convert_molar_energy_to_si(value: float, from_unit: str) -> float:
    """
    Converts molar energy (Enthalpy, Gibbs, Helmholtz, Internal Energy) 
    to the base SI unit Joules per mole (J/mol).
    
    Expected from_unit inputs: 'J/mol', 'kJ/mol', 'cal/mol', 'kcal/mol', 'BTU/lbmol'.
    The conversion is case-insensitive and ignores trailing/leading whitespaces.
    
    Note: Molar energy properties can mathematically and physically be negative. 
    Therefore, no strictly positive validation check is applied.
    
    Args:
        value (float): The molar energy value to convert.
        from_unit (str): The original unit of the molar energy.
        
    Returns:
        float: Molar energy in Joules per mole (J/mol).
        
    Raises:
        ValueError: If the unit is not recognized or supported.
    """
    # Sanitize the input to handle messy agent strings
    unit = from_unit.strip().lower()
    
    if unit in ['j/mol', 'joule/mol', 'joules/mol']:
        j_mol_value = value
    elif unit in ['kj/mol', 'kilojoule/mol', 'kilojoules/mol']:
        j_mol_value = value * 1000.0
    elif unit in ['cal/mol', 'calorie/mol', 'calories/mol']:
        # Thermochemical calorie
        j_mol_value = value * 4.184
    elif unit in ['kcal/mol', 'kilocalorie/mol', 'kilocalories/mol']:
        j_mol_value = value * 4184.0
    elif unit in ['btu/lbmol']:
        # 1 BTU = 1055.05585 J
        # 1 lb-mol = 453.59237 mol
        j_mol_value = value * (1055.05585 / 453.59237)
    else:
        raise ValueError(
            f"Unrecognized molar energy unit: '{from_unit}'. "
            "Supported units are: J/mol, kJ/mol, cal/mol, kcal/mol, BTU/lbmol."
        )
        
    return j_mol_value


def convert_molar_entropy_to_si(value: float, from_unit: str) -> float:
    """
    Converts molar entropy or heat capacity to the base SI unit Joules per mole-Kelvin (J/(mol*K)).
    
    Expected from_unit inputs: 'J/(mol*K)', 'kJ/(mol*K)', 'cal/(mol*K)', 'kcal/(mol*K)', 'BTU/(lbmol*R)'.
    The conversion is case-insensitive, ignores whitespaces, and accepts common syntax variations.
    
    Note: While absolute entropy is strictly positive, residual entropy (S^R) 
    and entropy changes (delta S) can physically be negative. Thus, no strictly 
    positive validation check is enforced here.
    
    Args:
        value (float): The molar entropy/capacity value to convert.
        from_unit (str): The original unit of the entropy.
        
    Returns:
        float: Molar entropy in Joules per mole-Kelvin (J/(mol*K)).
        
    Raises:
        ValueError: If the unit is not recognized or supported.
    """
    # Sanitize the input to handle messy agent strings (removes spaces to catch 'J / (mol * K)')
    unit = from_unit.strip().lower().replace(" ", "")
    
    if unit in ['j/(mol*k)', 'j/mol-k', 'j/molk', 'joule/(mol*k)']:
        j_mol_k_value = value
    elif unit in ['kj/(mol*k)', 'kj/mol-k', 'kj/molk', 'kilojoule/(mol*k)']:
        j_mol_k_value = value * 1000.0
    elif unit in ['cal/(mol*k)', 'cal/mol-k', 'cal/molk', 'calorie/(mol*k)']:
        # Thermochemical calorie
        j_mol_k_value = value * 4.184
    elif unit in ['kcal/(mol*k)', 'kcal/mol-k', 'kcal/molk', 'kilocalorie/(mol*k)']:
        j_mol_k_value = value * 4184.0
    elif unit in ['btu/(lbmol*r)', 'btu/lbmol-r', 'btu/lbmolr']:
        # 1 BTU = 1055.05585 J
        # 1 lb-mol = 453.59237 mol
        # 1 Rankine = 5/9 Kelvin
        j_mol_k_value = value * (1055.05585 / (453.59237 * (5.0 / 9.0)))
    else:
        raise ValueError(
            f"Unrecognized molar entropy unit: '{from_unit}'. "
            "Supported units are: J/(mol*K), kJ/(mol*K), cal/(mol*K), kcal/(mol*K), BTU/(lbmol*R)."
        )
        
    return j_mol_k_value


def convert_amount_to_moles(value: float, from_unit: str) -> float:
    """
    Converts substance amount to the base SI unit moles (mol).
    
    Expected from_unit inputs: 'mol', 'kmol', 'mmol', 'lbmol'.
    The conversion is case-insensitive and ignores trailing/leading whitespaces.
    
    Note: This function currently handles purely molar magnitude conversions. 
    Converting from mass (g, kg, lb) to moles requires the molar mass of the 
    specific fluid and is outside the scope of this baseline magnitude converter.
    
    Args:
        value (float): The amount value to convert.
        from_unit (str): The original unit of the substance amount.
        
    Returns:
        float: Amount in moles (mol).
        
    Raises:
        ValueError: If the unit is not recognized or supported.
        ValueError: If the resulting amount is less than or equal to zero.
    """
    # Sanitize the input to handle messy agent strings
    unit = from_unit.strip().lower()
    
    if unit in ['mol', 'mole', 'moles']:
        mol_value = value
    elif unit in ['kmol', 'kilomol', 'kilomole', 'kilomoles']:
        mol_value = value * 1000.0
    elif unit in ['mmol', 'millimol', 'millimole', 'millimoles']:
        mol_value = value * 1e-3
    elif unit in ['lbmol', 'lb-mol', 'pound-mole', 'pound-moles']:
        # 1 lb-mol is exactly 453.59237 moles
        mol_value = value * 453.59237
    else:
        raise ValueError(
            f"Unrecognized amount unit: '{from_unit}'. "
            "Supported units are: mol, kmol, mmol, lbmol."
        )
        
    # Thermodynamic reality check for amount of substance
    if mol_value <= 0:
        raise ValueError(
            f"Physically impossible state: Calculated amount ({mol_value:.5e} mol) "
            "must be strictly positive (n > 0) for EOS calculations."
        )
        
    return mol_value

conversion_tools_list = [
    convert_temperature_to_si,
    convert_pressure_to_si,
    convert_volume_to_si,
    convert_molar_volume_to_si,
    convert_energy_to_si,
    convert_molar_energy_to_si,
    convert_molar_entropy_to_si,
    convert_amount_to_moles
]