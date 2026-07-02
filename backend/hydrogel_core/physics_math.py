import numpy as np
from typing import Dict, Any, Tuple

class HydrogelPhysicsMath:
    """
    Pure mathematical and physical core engine for hydrogel simulation.
    Handles multi-scale modeling using vectorised NumPy operations.
    """
    def __init__(self, lcst_celsius: float = 32.0, empirical_constant_k: float = 150.0):
        """
        Initializes the thermodynamic and manufacturing constants.
        """
        self.lcst = lcst_celsius
        self.k_factor = empirical_constant_k

    def calculate_electrospinning_diameter(self, voltage_kv: float, viscosity_pa_s: float, flow_rate_ml_h: float) -> float:
        """
        [LEVEL 4 FEATURE] Computes the nominal nanofiber diameter using scaling laws.
        
        Args:
            voltage_kv (float): Applied electrical potential in kilovolts.
            viscosity_pa_s (float): Dynamic viscosity of the polymeric solution.
            flow_rate_ml_h (float): Solution feeding rate.
            
        Returns:
            float: Nanofiber diameter in nanometers (nm).
        """
        if voltage_kv <= 0:
            raise ValueError("Voltage must be strictly positive to establish an electrostatic jet.")
            
        # Analytical scaling law equation application
        dimensionless_ratio = (flow_rate_ml_h * viscosity_pa_s) / voltage_kv
        fiber_diameter_nm = self.k_factor * np.sqrt(dimensionless_ratio)
        
        return float(fiber_diameter_nm)

    def calculate_thermal_swelling_factor(self, temperature_celsius: float, alpha_sensitivity: float = 0.5) -> float:
        """
        [LEVEL 1 FEATURE] Models the volumetric thermal collapse of the hydrogel network.
        Uses a continuous sigmoidal transition profile around the LCST boundary.
        
        Args:
            temperature_celsius (float): Operating thermal state of the matrix.
            alpha_sensitivity (float): Steepness coefficient of the phase transition.
            
        Returns:
            float: Dimensionless swelling ratio (V_swollen / V_collapsed).
        """
        swelling_max = 3.5  # Fully hydrated state value
        swelling_min = 1.0  # Collapsed network state value
        
        # Logistic sigmoidal distribution mapping the hydrogel volume phase transition (HPVT)
        exponent = alpha_sensitivity * (temperature_celsius - self.lcst)
        swelling_factor = swelling_min + (swelling_max - swelling_min) / (1.0 + np.exp(exponent))
        
        return float(swelling_factor)

    def evaluate_coupled_system(self, voltage_kv: float, viscosity_pa_s: float, 
                                flow_rate_ml_h: float, temperature_celsius: float) -> Dict[str, float]:
        """
        Orchestrates the mathematical execution coupling structural parameters with thermodynamics.
        """
        # Execute manufacturing level matrix resolution
        fiber_diameter = self.calculate_electrospinning_diameter(voltage_kv, viscosity_pa_s, flow_rate_ml_h)
        
        # Execute thermodynamic level matrix resolution
        swelling_ratio = self.calculate_thermal_swelling_factor(temperature_celsius)
        
        return {
            "fiber_diameter_nm": round(fiber_diameter, 2),
            "swelling_ratio_dimensionless": round(swelling_ratio, 4)
        }

# =====================================================================
# STANDALONE MATHEMATICAL TESTING BENCH (SANITY CHECK)
# =====================================================================
if __name__ == "__main__":
    print("Testing pure mathematical functions under isolated parameters...")
    
    # Instantiate the physics engine core
    engine = HydrogelPhysicsMath()
    
    # Scenario A: Baseline standard execution validation
    results = engine.evaluate_coupled_system(
        voltage_kv=20.0, 
        viscosity_pa_s=1.2, 
        flow_rate_ml_h=0.5, 
        temperature_celsius=37.0  # Above LCST, matrix should be collapsed
    )
    print(f"   ➤ Physiological State Results (37°C): {results}")
    
    # Scenario B: Ambient temperature execution validation
    results_cold = engine.evaluate_coupled_system(
        voltage_kv=20.0, 
        viscosity_pa_s=1.2, 
        flow_rate_ml_h=0.5, 
        temperature_celsius=22.0  # Below LCST, matrix should be swollen
    )
    print(f"   ➤ Room Temperature Results (22°C): {results_cold}")