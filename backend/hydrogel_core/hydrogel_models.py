import numpy as np
from typing import Dict, Any

import numpy as np
from typing import Dict, Any, Union

class HydrogelVectorizedEngine:
    """
    High-performance multi-scale mathematical core for hydrogel properties.
    Designed to process both scalar values and vectorized NumPy arrays simultaneously.
    """
    def __init__(self, lcst_celsius: float = 32.0, scaling_constant_k: float = 150.0):
        """
        Initializes core thermodynamic and manufacturing macromolecular constants.
        """
        self.lcst = lcst_celsius
        self.k_factor = scaling_constant_k

    def compute_fiber_diameter(
        self, 
        voltage_kv: Union[float, np.ndarray], 
        viscosity_pa_s: Union[float, np.ndarray], 
        flow_rate_ml_h: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        [LEVEL 4] Computes nominal electrospun nanofiber diameters using scaling laws.
        Supports element-wise array operations via vectorized mechanics.
        """
        # Ensure voltage contains no zero or negative parameters to protect matrix division
        if np.any(voltage_kv <= 0):
            raise ValueError("Voltage threshold arrays must contain strictly positive values (> 0 kV).")
            
        # Mathematical scaling law executed at the hardware layer via NumPy
        operational_ratio = (flow_rate_ml_h * viscosity_pa_s) / voltage_kv
        fiber_diameter_nm = self.k_factor * np.sqrt(operational_ratio)
        
        return fiber_diameter_nm

    def compute_swelling_ratio(
        self, 
        temperature_celsius: Union[float, np.ndarray], 
        sensitivity_slope: float = 0.5
    ) -> Union[float, np.ndarray]:
        """
        [LEVEL 1] Models the sigmoidal thermal volume transition of the polymer network.
        Handles continuous phase changes dynamically over spatial arrays.
        """
        max_swelling = 3.5  # Swollen matrix state threshold
        min_swelling = 1.0  # Collapsed matrix state threshold
        
        # NumPy vectorized exponential handles single floats or massive arrays instantly
        thermal_deviation = sensitivity_slope * (temperature_celsius - self.lcst)
        swelling_factor = min_swelling + (max_swelling - min_swelling) / (1.0 + np.exp(thermal_deviation))
        
        return swelling_factor

    def execute_coupled_simulation(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates multi-scale execution pipelines mapping processing inputs to physical states.
        Enforces clean runtime array formatting for statistical data delivery.
        """
        voltages = np.array(telemetry["voltage_kv"], dtype=float)
        viscosities = np.array(telemetry["viscosity_pa_s"], dtype=float)
        flows = np.array(telemetry["flow_rate_ml_h"], dtype=float)
        temperatures = np.array(telemetry["temperature_celsius"], dtype=float)
        
        # Execute vectorized structural operations
        fiber_array = self.compute_fiber_diameter(voltages, viscosities, flows)
        swelling_array = self.compute_swelling_ratio(temperatures)
        
        return {
            "fiber_diameter_nm_mean": float(np.mean(fiber_array)),
            "fiber_diameter_nm_std": float(np.std(fiber_array)),
            "swelling_ratio_mean": float(np.mean(swelling_array)),
            "swelling_ratio_std": float(np.std(swelling_array)),
            "raw_arrays": {
                "fiber_diameter_nm": fiber_array.tolist() if fiber_array.ndim > 0 else float(fiber_array),
                "swelling_ratio": swelling_array.tolist() if swelling_array.ndim > 0 else float(swelling_array)
            }
        }


# =====================================================================
# MATHEMATICAL STANDALONE VERIFICATION BENCHMARK
# =====================================================================
if __name__ == "__main__":
    print("Initiating vectorized mathematical array calculations...")
    engine = HydrogelVectorizedEngine()
    
    # Simulating a multi-sample laboratory tracking run (Array Input)
    # This emulates 5 simultaneous experimental grid test points
    experimental_grid_payload = {
        "voltage_kv": [20.0, 21.5, 22.0, 22.5, 23.0],
        "viscosity_pa_s": [1.2, 1.25, 1.22, 1.28, 1.3],
        "flow_rate_ml_h": [0.5, 0.5, 0.52, 0.48, 0.5],
        "temperature_celsius": [37.0, 36.8, 37.2, 37.0, 36.9]
    }
    
    # Process array payload across scaling laws
    array_results = engine.execute_coupled_simulation(experimental_grid_payload)
    
    print("\n📊 [VECTORIZED SIMULATION METRICS GENERATED]:")
    print(f"   ➤ Fiber Diameter Mean: {array_results['fiber_diameter_nm_mean']:.2f} nm")
    print(f"   ➤ Fiber Diameter Std Dev: {array_results['fiber_diameter_nm_std']:.2f} nm")
    print(f"   ➤ Network Swelling Mean: {array_results['swelling_ratio_mean']:.4f}")