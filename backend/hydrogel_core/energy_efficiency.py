import numpy as np
from typing import Dict, Any, Union

class HydrogelEnergyEngine:
    """
    Mathematical physics core focused on Level 2: Energy Efficiency.
    Evaluates power overhead and Specific Energy Consumption (SEC) for manufacturing.
    """
    def __init__(self, solution_density_g_ml: float = 1.05):
        """
        Initializes the energetic engine with materials database baselines.
        """
        self.density = solution_density_g_ml

    def compute_power_consumption(
        self, 
        voltage_kv: Union[float, np.ndarray], 
        current_ua: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculates the net electrical power demanded by the high-voltage jet setup.
        Units: Output is resolved in Watts (W).
        """
        if np.any(voltage_kv < 0) or np.any(current_ua < 0):
            raise ValueError("Electrical telemetry parameters cannot possess negative values.")
            
        # P = V * I -> (kV * 10^3) * (uA * 10^-6) = Watts
        power_watts = (voltage_kv * 1000.0) * (current_ua * 1e-6)
        return power_watts

    def compute_specific_energy_consumption(
        self, 
        power_watts: Union[float, np.ndarray], 
        flow_rate_ml_h: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Computes the Specific Energy Consumption (SEC).
        Represents the energy footprint required to process a unit mass of nanofiber.
        Units: Output is resolved in kilowatt-hours per gram (kWh/g).
        """
        if np.any(flow_rate_ml_h <= 0):
            raise ValueError("Volumetric mass flow rate must be strictly positive (> 0 mL/h).")
            
        # Mass feed rate calculation: flow_rate (mL/h) * density (g/mL) = g/h
        mass_flow_g_h = flow_rate_ml_h * self.density
        
        # SEC = (Power in kW) / (Mass flow in g/h) = kWh/g
        power_kw = power_watts / 1000.0
        sec_kwh_g = power_kw / mass_flow_g_h
        
        return sec_kwh_g

    def execute_energetic_pipeline(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates Level 2 calculations handling vectorized matrix evaluations.
        Maps real-time electrical diagnostic arrays to processing efficiency summaries.
        """
        voltages = np.array(telemetry["voltage_kv"], dtype=float)
        currents = np.array(telemetry["current_ua"], dtype=float)
        flows = np.array(telemetry["flow_rate_ml_h"], dtype=float)
        
        # 1. Evaluate instant operational power steps
        calculated_power = self.compute_power_consumption(voltages, currents)
        
        # 2. Evaluate specific mass energy footprint requirements
        calculated_sec = self.compute_specific_energy_consumption(calculated_power, flows)
        
        return {
            "power_watts_mean": float(np.mean(calculated_power)),
            "power_watts_std": float(np.std(calculated_power)),
            "sec_kwh_g_mean": float(np.mean(calculated_sec)),
            "sec_kwh_g_std": float(np.std(calculated_sec)),
            "raw_energy_arrays": {
                "power_watts": calculated_power.tolist() if calculated_power.ndim > 0 else float(calculated_power),
                "sec_kwh_g": calculated_sec.tolist() if calculated_sec.ndim > 0 else float(calculated_sec)
            }
        }


# =====================================================================
# STANDALONE ENERGETIC CALCULATIONS VERIFICATION BENCHMARK
# =====================================================================
if __name__ == "__main__":
    print("Initializing Level 2 energy efficiency matrix validation checks...")
    energy_core = HydrogelEnergyEngine(solution_density_g_ml=1.08)
    
    # Mocking an automated 5-step voltage sweep profile from a laboratory dataset
    test_telemetry_batch = {
        "voltage_kv": [15.0, 18.0, 20.0, 22.0, 25.0],
        "current_ua": [0.8, 1.1, 1.3, 1.5, 1.9],
        "flow_rate_ml_h": [0.4, 0.45, 0.5, 0.5, 0.55]
    }
    
    # Process vectorized matrices through energy efficiency algorithms
    efficiency_summary = energy_core.execute_energetic_pipeline(test_telemetry_batch)
    
    print("\n⚡ [ENERGY METRICS GENERATED SUCCESSFULLY]:")
    print(f"   ➤ Average Power Draw: {efficiency_summary['power_watts_mean']:.3f} W")
    print(f"   ➤ Specific Energy Consumption (SEC) Mean: {efficiency_summary['sec_kwh_g_mean']:.5f} kWh/g")
    print(f"   ➤ SEC Standard Deviation: {efficiency_summary['sec_kwh_g_std']:.5f} kWh/g")