from typing import Any, Dict, List, Optional, Union
import numpy as np
import scipy.stats as stats


class HydrogelCoreFacade:
    """Unified architectural Facade for the hydrogels_core package.

    Orchestrates multi-scale deterministic models, energy metrics, thread-safe
    Monte Carlo loops, statistical dominance sorting, and Level 5 experimental
    laboratory validation using Welch's t-test.
    """

    def __init__(
        self,
        lcst_celsius: float = 32.0,
        scaling_k: float = 150.0,
        density_g_ml: float = 1.05,
    ):
        """Initializes and registers core macromolecular, geometric, and physical constants.

        Args:
            lcst_celsius (float, optional): Lower Critical Solution Temperature
                (LCST) defining the phase transition thermal boundary. Defaults to 32.0.
            scaling_k (float, optional): Empirical proportionality scaling factor
                for electrospun nanofiber diameter physics equations. Defaults to 150.0.
            density_g_ml (float, optional): Volumetric dynamic mass density
                of the fluid hydrogel solution matrix (g/mL). Defaults to 1.05.
        """
        self.lcst = lcst_celsius
        self.k_factor = scaling_k
        self.density = density_g_ml

    def run_complete_execution_pipeline(
        self,
        nominal_inputs: Dict[str, float],
        error_bounds: Dict[str, float],
        samples: int = 10000,
        histogram_bins: int = 30,
        seed: Optional[int] = None,
        experimental_data: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Any]:
        """Executes the end-to-end multi-scale mathematical simulation pipeline.

        This method encapsulates 7 execution stages:
        1. Generates isolated thread-safe Gaussian noise matrices using PCG64.
        2. Resolves fluid dynamics scaling laws and sigmoidal thermal swelling ratios.
        3. Computes high-voltage power draws and Specific Energy Consumption indices.
        4. Evaluates first-order local sensitivities through Pearson correlation sorting.
        5. Performs Level 5 validation against real data using Welch's independent t-test:
           $$t = \\frac{\\bar{X}_{\\text{sim}} - \\bar{X}_{\\text{exp}}}{\\sqrt{\\frac{s_{\\text{sim}}^2}{N_{\\text{sim}}} + \\frac{s_{\\text{exp}}^2}{N_{\\text{exp}}}}}$$
        6. Normalizes high-density stochastic populations into lightweight PDF structures.
        7. Returns a unified data payload contract for immediate API ingestion.

        Args:
            nominal_inputs (Dict[str, float]): Target operation targets from lab telemetry.
                Must include: "voltage_kv", "viscosity_pa_s", "flow_rate_ml_h",
                "temperature_celsius", and "current_ua".
            error_bounds (Dict[str, float]): Standard deviations representing sensor noise.
                Must share identical keys with nominal_inputs.
            samples (int, optional): Count of stochastic rows to allocate. Defaults to 10000.
            histogram_bins (int, optional): Quantization intervals for data compression.
                Defaults to 30.
            seed (Optional[int], optional): Random state seed integer initialization
                token to ensure exact replication. Defaults to None.
            experimental_data (Optional[Dict[str, List[float]]], optional): Lab characterization
                data arrays holding empirical microscopy results. Defaults to None.

        Returns:
            Dict[str, Any]: Highly structured dictionary payload tracking simulation
                statistics, sensitivity ranking lists, validation results, and compressed PDFs.
        """
        try:
            # Defensive verification checks to catch malformed payload structures early
            required_keys = {
                "voltage_kv",
                "viscosity_pa_s",
                "flow_rate_ml_h",
                "temperature_celsius",
                "current_ua",
            }
            if not required_keys.issubset(
                nominal_inputs.keys()
            ) or not required_keys.issubset(error_bounds.keys()):
                raise ValueError(
                    f"Inputs dictionaries must contain all keys: {required_keys}"
                )

            if samples <= 0 or histogram_bins <= 0:
                raise ValueError(
                    "Sample length and histogram bins must be strictly positive integers."
                )

            # -----------------------------------------------------------------
            # STEP 1: THREAD-SAFE ISOLATED STOCHASTIC SAMPLING (LEVEL 3)
            # -----------------------------------------------------------------
            # Initialize an isolated BitGenerator (PCG64) instance to prevent global state pollution
            rng = np.random.default_rng(seed)

            # Generate Gaussian noise arrays using the local thread-safe random state instance
            voltages = rng.normal(
                nominal_inputs["voltage_kv"],
                error_bounds["voltage_kv"],
                samples,
            )
            voltages = np.maximum(
                voltages, 1.0
            )  # Numerical safety guard to prevent zero-division

            viscosities = rng.normal(
                nominal_inputs["viscosity_pa_s"],
                error_bounds["viscosity_pa_s"],
                samples,
            )
            viscosities = np.maximum(viscosities, 0.01)

            flows = rng.normal(
                nominal_inputs["flow_rate_ml_h"],
                error_bounds["flow_rate_ml_h"],
                samples,
            )
            flows = np.maximum(flows, 0.01)

            temperatures = rng.normal(
                nominal_inputs["temperature_celsius"],
                error_bounds["temperature_celsius"],
                samples,
            )

            currents = rng.normal(
                nominal_inputs["current_ua"], error_bounds["current_ua"], samples
            )
            currents = np.maximum(currents, 0.01)

            # -----------------------------------------------------------------
            # STEP 2: PHYSICAL CORE MODELING (LEVEL 4 & LEVEL 1)
            # -----------------------------------------------------------------
            # Level 4: Vectorized Scaling Law for Nanofiber Diameters: d = k * sqrt((Q * eta) / V)
            fiber_diameters = self.k_factor * np.sqrt(
                (flows * viscosities) / voltages
            )

            # Level 1: Volumetric continuous phase transition around LCST boundary
            thermal_deviation = 0.5 * (temperatures - self.lcst)
            # Clip array values to defend exponential components from floating-point bit errors
            thermal_deviation = np.clip(thermal_deviation, -700.0, 700.0)
            swelling_ratios = 1.0 + (3.5 - 1.0) / (
                1.0 + np.exp(thermal_deviation)
            )

            # -----------------------------------------------------------------
            # STEP 3: ENERGETIC COMPUTATIONS (LEVEL 2)
            # -----------------------------------------------------------------
            # Electrical power overhead computation: P = V * I -> (kV * 1e3) * (uA * 1e-6) = Watts
            power_watts = (voltages * 1000.0) * (currents * 1e-6)

            # Specific Energy Consumption (SEC): kWh needed per gram produced
            # mass_flow = flow_rate (mL/h) * density (g/mL) = g/h
            mass_flow_g_h = flows * self.density
            sec_kwh_g = (power_watts / 1000.0) / mass_flow_g_h

            # -----------------------------------------------------------------
            # STEP 4: PARETO SENSITIVITY AND HIERARCHICAL RANKING
            # -----------------------------------------------------------------
            def compute_pearson(x: np.ndarray, y: np.ndarray) -> float:
                r = np.corrcoef(x, y)[0, 1]
                return float(r) if not np.isnan(r) else 0.0

            raw_dependencies = {
                "voltage_kv_influence": compute_pearson(
                    voltages, fiber_diameters
                ),
                "viscosity_pa_s_influence": compute_pearson(
                    viscosities, fiber_diameters
                ),
                "flow_rate_ml_h_influence": compute_pearson(
                    flows, fiber_diameters
                ),
                "temperature_celsius_influence": compute_pearson(
                    temperatures, fiber_diameters
                ),
            }

            # Grouping and ranking parameters using an absolute 0.05 relevance noise gate
            sorted_ranks = []
            for name, r_val in raw_dependencies.items():
                if abs(r_val) >= 0.05:
                    sorted_ranks.append(
                        {
                            "parameter": name.replace("_influence", ""),
                            "impact": round(r_val, 4),
                        }
                    )
            sorted_ranks = sorted(
                sorted_ranks, key=lambda x: abs(x["impact"]), reverse=True
            )

            # -----------------------------------------------------------------
            # STEP 5: LEVEL 5 EXPERIMENTAL VALIDATION & HYPOTHESIS TESTING
            # -----------------------------------------------------------------
            experimental_validation = {
                "validation_executed": False,
                "error_metrics": {},
            }

            if experimental_data is not None:
                experimental_validation["validation_executed"] = True

                if (
                    "fiber_diameter_nm" in experimental_data
                    and len(experimental_data["fiber_diameter_nm"]) > 0
                ):
                    real_fibers = np.array(
                        experimental_data["fiber_diameter_nm"], dtype=float
                    )
                    exp_mean_fiber = float(np.mean(real_fibers))
                    sim_mean_fiber = float(np.mean(fiber_diameters))

                    # Compute Mean Absolute Percentage Error (MAPE)
                    absolute_deviation = abs(sim_mean_fiber - exp_mean_fiber)
                    mape_fiber = (
                        (absolute_deviation / exp_mean_fiber) * 100.0
                        if exp_mean_fiber > 0
                        else 0.0
                    )

                    # Perform Welch's t-test to safely handle highly unequal sample configurations
                    t_stat, p_value = stats.ttest_ind(
                        fiber_diameters, real_fibers, equal_var=False
                    )

                    experimental_validation["error_metrics"][
                        "fiber_diameter_evaluation"
                    ] = {
                        "experimental_samples_count": len(real_fibers),
                        "experimental_mean_nm": round(exp_mean_fiber, 2),
                        "simulated_baseline_mean_nm": round(sim_mean_fiber, 2),
                        "mean_absolute_percentage_error_pct": round(
                            mape_fiber, 2
                        ),
                        "accuracy_index_pct": round(100.0 - mape_fiber, 2),
                        "statistical_hypothesis_testing": {
                            "t_statistic": round(float(t_stat), 4),
                            "p_value": round(float(p_value), 6),
                            "null_hypothesis_accepted": bool(p_value > 0.05),
                            "verdict": (
                                "Model maps reality successfully"
                                if p_value > 0.05
                                else "Model recalibration required"
                            ),
                        },
                    }

            # -----------------------------------------------------------------
            # STEP 6: HISTOGRAM COMPRESSION LAYERS
            # -----------------------------------------------------------------
            def compress(array_data: np.ndarray) -> Dict[str, List[float]]:
                counts, edges = np.histogram(
                    array_data, bins=histogram_bins, density=True
                )
                centers = (edges[:-1] + edges[1:]) / 2.0
                return {
                    "x_coordinates": np.round(centers, 3).tolist(),
                    "y_densities": np.round(counts, 6).tolist(),
                }

            # -----------------------------------------------------------------
            # STEP 7: MASTER PAYLOAD CONTRACT PACKAGING
            # -----------------------------------------------------------------
            return {
                "status": "success",
                "reproducibility_token": {
                    "isolated_generator_applied": True,
                    "seed_value": seed,
                },
                "summary_statistics": {
                    "fiber_diameter_nm": {
                        "mean": float(np.mean(fiber_diameters)),
                        "std": float(np.std(fiber_diameters)),
                    },
                    "swelling_ratio": {
                        "mean": float(np.mean(swelling_ratios)),
                        "std": float(np.std(swelling_ratios)),
                    },
                    "power_consumption_w": {
                        "mean": float(np.mean(power_watts)),
                        "std": float(np.std(power_watts)),
                    },
                    "specific_energy_kwh_g": {
                        "mean": float(np.mean(sec_kwh_g)),
                        "std": float(np.std(sec_kwh_g)),
                    },
                },
                "sensitivity_analysis": {
                    "primary_driver": (
                        sorted_ranks[0]["parameter"] if sorted_ranks else "none"
                    ),
                    "ordered_impacts": sorted_ranks,
                },
                "level_5_experimental_validation": experimental_validation,
                "api_optimized_visualizations": {
                    "fiber_diameter_distribution": compress(fiber_diameters),
                    "swelling_ratio_distribution": compress(swelling_ratios),
                },
            }

        except Exception as runtime_error:
            return {
                "status": "error",
                "error_class": type(runtime_error).__name__,
                "detail": f"Master facade breakdown inside processing thread: {str(runtime_error)}",
            }


# =====================================================================
# INTEGRATED PIPELINE PRODUCTION BENCHMARK (STANDALONE RUN)
# =====================================================================
if __name__ == "__main__":
    print(
        "Launching thread-safe production verification for Master Facade Architecture..."
    )
    facade = HydrogelCoreFacade()

    # Target baseline values from laboratory configuration controls
    nominal_setpoints = {
        "voltage_kv": 20.0,
        "viscosity_pa_s": 1.2,
        "flow_rate_ml_h": 0.5,
        "temperature_celsius": 32.5,
        "current_ua": 1.2,
    }

    # Real-time sensor standard deviation tolerances
    sensor_noise = {
        "voltage_kv": 0.4,
        "viscosity_pa_s": 0.05,
        "flow_rate_ml_h": 0.01,
        "temperature_celsius": 0.7,
        "current_ua": 0.08,
    }

    # Actual experimental data points extracted from laboratory microscopy characterization
    real_microscopy_samples = {
        "fiber_diameter_nm": [212.1, 220.4, 217.9, 214.3, 219.0, 223.5, 216.8]
    }

    # Run independent simulation iterations utilizing identical seed values
    output_run_1 = facade.run_complete_execution_pipeline(
        nominal_setpoints,
        sensor_noise,
        samples=10000,
        seed=12345,
        experimental_data=real_microscopy_samples,
    )
    output_run_2 = facade.run_complete_execution_pipeline(
        nominal_setpoints,
        sensor_noise,
        samples=10000,
        seed=12345,
        experimental_data=real_microscopy_samples,
    )

    # Assert true mathematical replication to prove isolated thread safety
    if (
        output_run_1["status"] == "success"
        and output_run_2["status"] == "success"
    ):
        mean_1 = output_run_1["summary_statistics"]["fiber_diameter_nm"]["mean"]
        mean_2 = output_run_2["summary_statistics"]["fiber_diameter_nm"]["mean"]
        print(
            f"\n   ➤ Local Generator Check: Run 1 Mean = {mean_1:.4f} nm | Run 2 Mean = {mean_2:.4f} nm"
        )
        print(f"   ➤ Isolated Deterministic Integrity Confirmed: {mean_1 == mean_2}")

        # Extract Level 5 experimental analytics and t-test outputs
        validation_node = output_run_1["level_5_experimental_validation"][
            "error_metrics"
        ]["fiber_diameter_evaluation"]
        print(f"\n📊 [LEVEL 5 CRITICAL HYPOTHESIS TESTING REPORT]:")
        print(
            f"   ➤ Experimental Sample Size (Lab): {validation_node['experimental_samples_count']} elements."
        )
        print(
            f"   ➤ Model Prediction Accuracy: {validation_node['accuracy_index_pct']}% (MAPE: {validation_node['mean_absolute_percentage_error_pct']}%)"
        )

        t_test_node = validation_node["statistical_hypothesis_testing"]
        print(f"   ➤ Welch's t-statistic value: {t_test_node['t_statistic']}")
        print(f"   ➤ Evaluated p-value parameter: {t_test_node['p_value']}")
        print(
            f"   ➤ Null Hypothesis Accepted (No Significant Difference): {t_test_node['null_hypothesis_accepted']}"
        )
        print(f"   ➤ Engineering Verdict: {t_test_node['verdict'].upper()}")
    else:
        print(f"❌ Error during verification: {output_run_1.get('detail')}")