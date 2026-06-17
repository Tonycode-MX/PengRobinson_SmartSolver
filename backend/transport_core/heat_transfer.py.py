import numpy as np
import CoolProp.CoolProp as CP
from typing import Dict, Any, Union

class ThermalAnalysisTools:
    """
    Thermal analysis toolkit designed specifically for AI Agent consumption.
    All functions act as 'Tools' and return structured responses (JSON-like dictionaries).
    Strictly avoids prints() or sys.exit() to prevent breaking the LLM's thought loop.
    """

    @staticmethod
    def calculate_gas_radiation(
        gas: str, 
        T_gas_R: float, 
        T_surf_R: float, 
        partial_pressure_atm: float, 
        geometry: str, 
        char_dimension_ft: float, 
        area_ft2: float
    ) -> Dict[str, Union[float, str, int]]:
        """
        [TOOL] Calculates thermal radiation heat transfer for asymmetric/radiant gases.
        
        Utility:
            Use this tool ONLY when analyzing radiant gases (specifically Carbon Dioxide "CO2" 
            or Water Vapor "H2O"). It computes the equivalent beam length based on container geometry, 
            estimates gas emissivity using empirical logarithmic models, and calculates total 
            heat transfer via the Stefan-Boltzmann law.
        
        Args:
            gas (str): The radiant gas identifier. Must be strictly "CO2" or "H2O".
            T_gas_R (float): Gas temperature in degrees Rankine (°R).
            T_surf_R (float): Surface/wall temperature in degrees Rankine (°R).
            partial_pressure_atm (float): Partial pressure of the radiant gas in atmospheres (atm).
            geometry (str): Container shape. Must be "sphere", "cylinder", or "plates".
            char_dimension_ft (float): Characteristic dimension (diameter or plate spacing) in feet (ft).
            area_ft2 (float): Surface area for heat transfer in square feet (ft^2).
            
        Returns:
            Dict: A structured dictionary containing:
                - status (int): HTTP-like status code (200 for success).
                - gas (str): The processed gas name.
                - emissivity (float): Dimensionless calculated gas emissivity.
                - q_radiation_btu_h (float): Total radiation heat transfer in Btu/h.
                - beam_length_ft (float): Equivalent beam length (L) in feet.
                
        Error Validations:
            - Returns status 400 if an unsupported 'geometry' is provided.
            - Returns status 400 if an unsupported 'gas' is provided.
            - Returns status 500 if an internal mathematical error occurs.
        """
        try:
            gas = gas.strip().upper()
            geometry = geometry.strip().lower()
            
            # 1. Geometric Analysis
            geometries = {"sphere": 0.65, "cylinder": 0.90, "plates": 1.80}
            if geometry not in geometries:
                return {
                    "status": 400, 
                    "error": f"Invalid geometry '{geometry}'. Must be: 'sphere', 'cylinder', or 'plates'."
                }
                
            L = geometries[geometry] * char_dimension_ft
            pL = partial_pressure_atm * L
            pL_safe = float(np.maximum(pL, 0.0001)) # Prevents log(0) errors
            
            # 2. Emissivity Analysis
            if gas == "CO2":
                eps = 0.12 * np.log10(1 + pL_safe * 20) * (T_gas_R / 1000)**-0.6
            elif gas == "H2O":
                eps = 0.15 * np.log10(1 + pL_safe * 15) * (T_gas_R / 1000)**-0.8
            else:
                return {
                    "status": 400, 
                    "error": f"Gas '{gas}' not supported for radiation. Use 'CO2' or 'H2O'."
                }

            emissivity = float(np.clip(eps, 0.001, 1.0))
            
            # 3. Thermal Transfer Calculation (Stefan-Boltzmann)
            sigma = 0.1714e-8
            q_rad = area_ft2 * sigma * emissivity * (T_gas_R**4 - T_surf_R**4)
            
            return {
                "status": 200,
                "gas": gas,
                "emissivity": round(emissivity, 4),
                "q_radiation_btu_h": round(q_rad, 2),
                "beam_length_ft": round(L, 3)
            }
            
        except Exception as e:
            return {"status": 500, "error": f"Internal error during radiation calculation: {str(e)}"}

    @staticmethod
    def get_gas_properties(
        gas: str, 
        T_gas_R: float, 
        total_pressure_atm: float
    ) -> Dict[str, Union[float, str, int]]:
        """
        [TOOL] Extracts thermophysical properties for transparent/symmetric gases via CoolProp.
        
        Utility:
            Use this tool when analyzing transparent gases (e.g., Nitrogen, Oxygen, Helium) 
            that transfer heat primarily through convection and conduction, ignoring radiation.
            It acts as a bridge to the CoolProp backend, handling necessary unit conversions 
            from Imperial to SI units internally.
        
        Args:
            gas (str): Gas name in English compatible with CoolProp (e.g., "N2", "Helium", "Argon").
            T_gas_R (float): Gas temperature in degrees Rankine (°R).
            total_pressure_atm (float): Total system pressure in atmospheres (atm).
            
        Returns:
            Dict: A structured dictionary containing:
                - status (int): HTTP-like status code (200 for success).
                - gas (str): The properly formatted gas name.
                - thermal_conductivity_w_mk (float): Thermal conductivity (k) in W/m·K.
                - dynamic_viscosity_pa_s (float): Dynamic viscosity (μ) in Pa·s.
                - specific_heat_j_kgk (float): Specific heat at constant pressure (Cp) in J/kg·K.
                - density_kg_m3 (float): Density (ρ) in kg/m³.
                
        Error Validations:
            - Returns status 404 if CoolProp cannot recognize the gas name.
            - Returns status 500 if physical state calculations fail (e.g., impossible phase constraints).
        """
        try:
            # Mandatory unit conversions for CoolProp (Imperial to SI)
            T_K = T_gas_R * (5/9)
            P_Pa = total_pressure_atm * 101325
            gas = gas.strip().capitalize()
            
            # Using uppercase for chemical formulas (e.g., 'N2', 'O2')
            if gas.lower() in ["n2", "o2", "co2", "h2o"]:
                gas = gas.upper()
            
            return {
                "status": 200,
                "gas": gas,
                "thermal_conductivity_w_mk": float(CP.PropsSI('L', 'T', T_K, 'P', P_Pa, gas)),
                "dynamic_viscosity_pa_s": float(CP.PropsSI('V', 'T', T_K, 'P', P_Pa, gas)),
                "specific_heat_j_kgk": float(CP.PropsSI('C', 'T', T_K, 'P', P_Pa, gas)),
                "density_kg_m3": float(CP.PropsSI('D', 'T', T_K, 'P', P_Pa, gas))
            }
            
        except ValueError:
            return {
                "status": 404, 
                "error": f"CoolProp did not recognize the gas '{gas}'. Please verify spelling and capitalization."
            }
        except Exception as e:
            return {"status": 500, "error": f"Failure during property extraction: {str(e)}"}

    @staticmethod
    def route_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        [TOOL] Main router for the AI Agent. Ingests raw JSON data and delegates to the correct physics engine.
        
        Utility:
            This is the primary endpoint for the agent. Instead of the agent deciding which 
            sub-function to call, it passes all gathered context here. The function parses 
            the gas type and routes to either radiation or pure convection logic.
            
        Args:
            data (Dict[str, Any]): Raw dictionary containing all extracted prompt parameters.
                Expected keys: 'gas', 'T_gas_R', 'T_surf_R' (if radiation), 'geometry' (if radiation), 
                'char_dimension_ft' (if radiation), 'area_ft2' (if radiation), 'partial_pressure_atm', 
                'total_pressure_atm'.
                
        Returns:
            Dict: The structured output from the respective delegated tool.
            
        Error Validations:
            - Returns status 400 if a mandatory key is missing from the payload.
            - Returns status 400 if a string is provided where a float is required.
        """
        requested_gas = data.get("gas", "").strip().upper()
        radiant_gases = ["CO2", "H2O"]
        
        try:
            if requested_gas in radiant_gases:
                return ThermalAnalysisTools.calculate_gas_radiation(
                    gas=requested_gas,
                    T_gas_R=float(data["T_gas_R"]),
                    T_surf_R=float(data["T_surf_R"]),
                    partial_pressure_atm=float(data.get("partial_pressure_atm", 0.1)),
                    geometry=str(data["geometry"]),
                    char_dimension_ft=float(data["char_dimension_ft"]),
                    area_ft2=float(data["area_ft2"])
                )
            else:
                return ThermalAnalysisTools.get_gas_properties(
                    gas=requested_gas,
                    T_gas_R=float(data["T_gas_R"]),
                    total_pressure_atm=float(data.get("total_pressure_atm", 1.0))
                )
        except KeyError as missing_key:
            return {
                "status": 400, 
                "error": f"Missing mandatory payload parameter: {missing_key}"
            }
        except ValueError:
            return {
                "status": 400,