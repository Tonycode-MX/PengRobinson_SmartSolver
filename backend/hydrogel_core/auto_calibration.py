import numpy as np
import scipy.stats as stats
from typing import Dict, Any, List, Optional, Tuple

class HydrogelAutocalibrationEngine:
    """
    Advanced physical-mathematical core featuring adaptive Level 5 validation.
    Monitors Welch's t-test rejections and dynamically recalibrates empirical 
    scaling constants using mathematical expectation deviations.
    """
    def __init__(self, lcst_celsius: float = 32.0, initial_k: float = 150.0, density_g_ml: float = 1.05):
        """
        Initializes the adaptive engine with operational baselines and calibration logs.
        """
        self.lcst = lcst_celsius
        self.k_factor = initial_k
        self.density = density_g_ml
        self.calibration_history: List[Dict[str, Any]] = []

    def run_adaptive_pipeline(
        self, 
        nominal_inputs: Dict[str, float], 
        error_bounds: Dict[str, float], 
        samples: int = 10000,
        experimental_data: Optional[Dict[str, List[float]]] = None,
        auto_adjust: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the stochastic simulation loop and performs online self-calibration 
        if experimental data triggers statistical divergence boundaries.
        """
        # Initialize an isolated thread-safe random state instance
        rng = np.random.default_rng()
        
        # Stochastic sampling grid execution
        voltages = rng.normal(nominal_inputs["voltage_kv"], error_bounds["voltage_kv"], samples)
        voltages = np.maximum(voltages, 1.0)
        viscosities = rng.normal(nominal_inputs["viscosity_pa_s"], error_bounds["viscosity_pa_s"], samples)
        viscosities = np.maximum(viscosities, 0.01)
        flows = rng.normal(nominal_inputs["flow_rate_ml_h"], error_bounds["flow_rate_ml_h"], samples)
        flows = np.maximum(flows, 0.01)
        temperatures = rng.normal(nominal_inputs["temperature_celsius"], error_bounds["temperature_celsius"], samples)
        
        # Level 4 & Level 1 vectorized physics execution
        fiber_diameters = self.k_factor * np.sqrt((flows * viscosities) / voltages)
        thermal_deviation = 0.5 * (temperatures - self.lcst)
        swelling_ratios = 1.0 + (3.5 - 1.0) / (1.0 + np.exp(thermal_deviation))

        sim_mean_fiber = float(np.mean(fiber_diameters))
        sim_std_fiber = float(np.std(fiber_diameters))

        # Level 5 validation module closure
        validation_report = {"status": "no_experimental_data_provided", "recalibrated": False}

        if experimental_data and "fiber_diameter_nm" in experimental_data:
            real_fibers = np.array(experimental_data["fiber_diameter_nm"], dtype=float)
            exp_mean_fiber = float(np.mean(real_fibers))
            
            # Perform Welch's t-test to evaluate structural variance identity
            t_stat, p_value = stats.ttest_ind(fiber_diameters, real_fibers, equal_var=False)
            null_hypothesis_accepted = bool(p_value > 0.05)
            
            mape = (abs(sim_mean_fiber - exp_mean_fiber) / exp_mean_fiber) * 100.0
            original_k = self.k_factor
            
            # Trigger autocalibration if the model significantly deviates from reality
            if not null_hypothesis_accepted and auto_adjust:
                # Direct proportionality scaling law tuning: k_new = k_old * (mean_exp / mean_sim)
                correction_factor = exp_mean_fiber / sim_mean_fiber
                self.k_factor = float(original_k * correction_factor)
                
                validation_report["recalibrated"] = True
                validation_report["calibration_shift"] = {
                    "previous_k_factor": round(original_k, 4),
                    "updated_k_factor": round(self.k_factor, 4),
                    "scaling_ratio": round(float(correction_factor), 4)
                }
            
            validation_report["status"] = "success" if null_hypothesis_accepted else "recalibrated_due_to_divergence"
            validation_report["statistical_metrics"] = {
                "p_value": round(float(p_value), 6),
                "null_hypothesis_accepted": null_hypothesis_accepted,
                "mape_pct": round(mape, 2),
                "accuracy_pct": round(100.0 - mape, 2),
                "experimental_target_mean_nm": round(exp_mean_fiber, 2),
                "simulated_baseline_mean_nm": round(sim_mean_fiber, 2)
            }
            
            # Append current session metrics to execution historical registry tracking
            self.calibration_history.append({
                "timestamp_index": len(self.calibration_history) + 1,
                "p_value": float(p_value),
                "mape": mape,
                "adjusted": validation_report["recalibrated"],
                "final_k_value": self.k_factor
            })

        return {
            "engine_status": {
                "current_operational_k_factor": round(self.k_factor, 4),
                "total_calibration_cycles_logged": len(self.calibration_history)
            },
            "summary_statistics": {
                "fiber_diameter_nm": {"mean": round(sim_mean_fiber, 2), "std": round(sim_std_fiber, 2)},
                "swelling_ratio": {"mean": round(float(np.mean(swelling_ratios)), 4)}
            },
            "level_5_validation": validation_report
        }


# =====================================================================
# SYSTEM AUTOMATION VERIFICATION TESTING (ONLINE BENCHMARK)
# =====================================================================
if __name__ == "__main__":
    print("Testing Hydrogel Online Autocalibration System...")
    adaptive_core = HydrogelAutocalibrationEngine(initial_k=150.0)
    
    setpoints = {"voltage_kv": 20.0, "viscosity_pa_s": 1.2, "flow_rate_ml_h": 0.5, "temperature_celsius": 37.0}
    noise = {"voltage_kv": 0.4, "viscosity_pa_s": 0.05, "flow_rate_ml_h": 0.01, "temperature_celsius": 0.5}
    
    # Real microscopy data showing fibers are significantly larger than initial predictions
    microscope_batch = {"fiber_diameter_nm": [215.4, 222.1, 219.8, 228.3, 211.0, 220.5]}
    
    print(f"\n[CYCLE 1] Running baseline simulation with initial k = {adaptive_core.k_factor}...")
    run_1 = adaptive_core.run_adaptive_pipeline(setpoints, noise, experimental_data=microscope_batch)
    print(f"   ➤ Hypothesis Accepted? {run_1['level_5_validation']['statistical_metrics']['null_hypothesis_accepted']}")
    print(f"   ➤ Status Node: {run_1['level_5_validation']['status']}")
    print(f"   ➤ New Configured K Factor: {adaptive_core.k_factor:.2f}")
    
    print(f"\n[CYCLE 2] Verifying model accuracy after automatic tuning deployment...")
    run_2 = adaptive_core.run_adaptive_pipeline(setpoints, noise, experimental_data=microscope_batch)
    metrics_2 = run_2['level_5_validation']['statistical_metrics']
    print(f"   ➤ Hypothesis Accepted on Session 2? {metrics_2['null_hypothesis_accepted']}")
    print(f"   ➤ New Prediction Accuracy Index: {metrics_2['accuracy_pct']}%")
    print(f"   ➤ Absolute Target Mean: {metrics_2['experimental_target_mean_nm']} nm | New Simulated Mean: {metrics_2['simulated_mean_nm']} nm")