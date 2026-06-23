from langchain_core.tools import tool
from utils.conversions import convert_temperature_to_si, convert_pressure_to_si

@tool
def tool_convert_temperature(value: float, from_unit: str) -> float:
    """AI TOOL: Converts temperature to Kelvin (K). Valid from_unit strings: 'C', 'Celsius', 'F', 'Fahrenheit', 'R', 'Rankine', 'K'."""
    return convert_temperature_to_si(value, from_unit)

@tool
def tool_convert_pressure(value: float, from_unit: str) -> float:
    """AI TOOL: Converts absolute pressure to Pascals (Pa). Valid from_unit strings: 'atm', 'bar', 'psi', 'kPa', 'MPa', 'mmHg', 'Torr', 'Pa'."""
    return convert_pressure_to_si(value, from_unit)

# Exportamos la lista de herramientas de conversión
conversion_tools_list = [
    tool_convert_temperature,
    tool_convert_pressure
]