import numpy as np
from typing import Dict, Any

class HydrogelSensitivityAnalyzer:
    """
    Performs first-order local sensitivity analysis on hydrogel multi-scale models.
    Utilizes Pearson correlation matrices derived from high-speed Monte Carlo distributions.
    """
    def __init__(self, lcst_celsius: float = 32.0, scaling_constant_k: float = 150.0):
        """
        Initializes core physical properties for matrix mapping.
        """
        self.lcst = lcst_celsius
        self.k_factor = scaling_constant_k

    def compute_pearson_coefficient(self, x_input: np.ndarray, y_output: np.ndarray) -> float:
        """
        Calculates the standard mathematical Pearson correlation coefficient (r).
        Normalizes covariance against standard deviations.
        """
        correlation_matrix = np.corrcoef(x_input, y_output)
        # Extract index [0, 1] representing the cross-correlation value between arrays
        pearson_r = correlation_matrix[0, 1]
        
        # Guard against NaN values resulting from zero-variance simulations
        if np.isnan(pearson_r):
            return 0.0
            
        return float(pearson_r)

    def analyze_system_sensitivity(
        self, 
        nominal_inputs: Dict[str, float], 
        error_bounds: Dict[str, float], 
        samples: int = 10000
    ) -> Dict[str, Any]:
        """
        Executes a localized stochastic sweep and maps the directional sensitivity 
        of every operational telemetry input over structural outputs.
        """
        # 1. Generate normal stochastic sample arrays representing sensor noise
        voltage_arr = np.random.normal(nominal_inputs["voltage_kv"], error_bounds["voltage_kv"], samples)
        viscosity_arr = np.random.normal(nominal_inputs["viscosity_pa_s"], error_bounds["viscosity_pa_s"], samples)
        flow_arr = np.random.normal(nominal_inputs["flow_rate_ml_h"], error_bounds["flow_rate_ml_h"], samples)
        temp_arr = np.random.normal(nominal_inputs["temperature_celsius"], error_bounds["temperature_celsius"], samples)

        # Enforce physical boundaries to prevent square root or exponential runtime overflows
        voltage_arr = np.maximum(voltage_arr, 1.0)
        viscosity_arr = np.maximum(viscosity_arr, 0.01)
        flow_arr = np.maximum(flow_arr, 0.01)

        # 2. Execute Level 4 and Level 1 physical core equations simultaneously
        fiber_diameters = self.k_factor * np.sqrt((flow_arr * viscosity_arr) / voltage_arr)
        
        thermal_deviation = 0.5 * (temp_arr - self.lcst)
        swelling_ratios = 1.0 + (3.5 - 1.0) / (1.0 + np.exp(thermal_deviation))

        # 3. Map cross-correlations to create the final sensitivity matrix
        sensitivity_matrix = {
            "fiber_diameter_dependencies": {
                "voltage_kv_influence": self.compute_pearson_coefficient(voltage_arr, fiber_diameters),
                "viscosity_pa_s_influence": self.compute_pearson_coefficient(viscosity_arr, fiber_diameters),
                "flow_rate_ml_h_influence": self.compute_pearson_coefficient(flow_arr, fiber_diameters),
                "temperature_celsius_influence": self.compute_pearson_coefficient(temp_arr, fiber_diameters)
            },
            "swelling_ratio_dependencies": {
                "voltage_kv_influence": self.compute_pearson_coefficient(voltage_arr, swelling_ratios),
                "viscosity_pa_s_influence": self.compute_pearson_coefficient(viscosity_arr, swelling_ratios),
                "flow_rate_ml_h_influence": self.compute_pearson_coefficient(flow_arr, swelling_ratios),
                "temperature_celsius_influence": self.compute_pearson_coefficient(temp_arr, swelling_ratios)
            }
        }

        return {
            "status": "success",
            "samples_analyzed": samples,
            "sensitivity_indices": sensitivity_matrix,
            "raw_distribution_data_pures": {
                "fiber_diameter_nm": fiber_diameters.tolist(),
                "swelling_ratio": swelling_ratios.tolist()
            }
        }


# =====================================================================
# STANDALONE MODEL ANALYSIS EXAMPLES (VERIFICATION CHECK)
# =====================================================================
if __name__ == "__main__":
    print("Evaluating Pearson sensitivity matrix configurations...")
    analyzer = HydrogelSensitivityAnalyzer()
    
    # Standard engineering setpoints
    setpoints = {
        "voltage_kv": 20.0,
        "viscosity_pa_s": 1.2,
        "flow_rate_ml_h": 0.5,
        "temperature_celsius": 32.5  # Critical LCST phase transition zone
    }
    
    # Observed hardware sensor standard deviation limits
    deviations = {
        "voltage_kv": 0.6,
        "viscosity_pa_s": 0.08,
        "flow_rate_ml_h": 0.03,
        "temperature_celsius": 1.0
    }
    
    # Run the computation over 10,000 localized variations
    analysis_report = analyzer.analyze_system_sensitivity(setpoints, deviations, samples=10000)
    
    print("\n📈 [SENSITIVITY INDICES MATRIX RESULTS]:")
    fiber_deps = analysis_report["sensitivity_indices"]["fiber_diameter_dependencies"]
    print(f"   ➤ Voltage influence on Fiber Size: {fiber_deps['voltage_kv_influence']:.4f} (Inverted directionality)")
    print(f"   ➤ Viscosity influence on Fiber Size: {fiber_deps['viscosity_pa_s_influence']:.4f} (Direct relationship)")
    
    swelling_deps = analysis_report["sensitivity_indices"]["swelling_ratio_dependencies"]
    print(f"   ➤ Temperature influence on Hydrogel Hinchamiento: {swelling_deps['temperature_celsius_influence']:.4f} (Strong negative shift)")