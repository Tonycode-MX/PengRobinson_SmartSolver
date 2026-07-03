from typing import Any, Dict, Tuple
import numpy as np


class HydrogelSensitivityAnalyzer:
    """Performs first-order local sensitivity analysis on hydrogel multi-scale models.

    Utilizes Pearson correlation matrices derived from high-speed Monte Carlo
    distributions to establish linear coupling dependencies between inputs and outputs.
    """

    def __init__(
        self, lcst_celsius: float = 32.0, scaling_constant_k: float = 150.0
    ):
        """Initializes core physical properties for matrix mapping.

        Args:
            lcst_celsius (float, optional): Lower Critical Solution Temperature
                (LCST) in Celsius. Defaults to 32.0.
            scaling_constant_k (float, optional): Empirical proportionality
                factor for nanofiber scaling. Defaults to 150.0.
        """
        self.lcst = lcst_celsius
        self.k_factor = scaling_constant_k

    def compute_pearson_coefficient(
        self, x_input: np.ndarray, y_output: np.ndarray
    ) -> float:
        """Calculates the standard mathematical Pearson correlation coefficient ($r$).

        Normalizes the covariance of the two variables against the product of their
        standard deviations:
        $$r = \\frac{\\text{cov}(X, Y)}{\\sigma_X \\cdot \\sigma_Y} = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum_{i=1}^{n} (x_i - \\bar{x})^2 \\cdot \\sum_{i=1}^{n} (y_i - \\bar{y})^2}}$$

        Value updates fall strictly within $[-1.0, 1.0]$, where:
        - $r = 1.0$: Perfect direct linear relationship.
        - $r = -1.0$: Perfect inverse linear relationship.
        - $r = 0.0$: No linear correlation between datasets.

        Args:
            x_input (np.ndarray): Array representing the independent stochastic variable.
            y_output (np.ndarray): Array representing the dependent model output response.

        Returns:
            float: Computed Pearson linear correlation coefficient. Returns 0.0
                if indeterminate due to zero-variance input arrays.
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
        samples: int = 10000,
    ) -> Dict[str, Any]:
        """Executes a localized stochastic sweep and maps the directional sensitivity indices.

        Evaluates how fluctuations inside process parameters transmit variance into
        nanofiber dimensions and swelling dynamics by pairing Monte Carlo generation
        with automated Pearson matrix evaluation.

        Physical dependencies resolved:
        - Fiber Diameter ($d$): Derived from scaling law $d \\propto \\sqrt{\\frac{Q \\cdot \\eta}{V}}$.
        - Swelling Ratio ($S_f$): Derived from sigmoidal phase collapse curve.

        Args:
            nominal_inputs (Dict[str, float]): Base operational parameter setpoints.
                Required keys: "voltage_kv", "viscosity_pa_s", "flow_rate_ml_h",
                and "temperature_celsius".
            error_bounds (Dict[str, float]): Simulated Gaussian noise standard deviations.
                Must mirror keys present within nominal_inputs. Values must be non-negative.
            samples (int, optional): Size of the sample matrix array allocation.
                Must be a positive integer. Defaults to 10000.

        Returns:
            Dict[str, Any]: Compiled data structure storing results:
                - "status": Execution confirmation string.
                - "samples_analyzed": Sample iteration ceiling limit.
                - "sensitivity_indices": Nested dictionaries containing structural influence factors.
                - "raw_distribution_data_pures": Generated matrix outputs converted to native lists.

        Raises:
            ValueError: If sample iteration length is invalid or if target dictionary
                keys are missing or incorrect.
        """
        if samples <= 0:
            raise ValueError("Sample metrics must be strictly positive.")

        required_keys = {
            "voltage_kv",
            "viscosity_pa_s",
            "flow_rate_ml_h",
            "temperature_celsius",
        }
        if not required_keys.issubset(
            nominal_inputs.keys()
        ) or not required_keys.issubset(error_bounds.keys()):
            raise ValueError(
                f"Missing keys in telemetry inputs. Expected: {required_keys}"
            )

        # 1. Generate normal stochastic sample arrays representing sensor noise
        voltage_arr = np.random.normal(
            nominal_inputs["voltage_kv"], error_bounds["voltage_kv"], samples
        )
        viscosity_arr = np.random.normal(
            nominal_inputs["viscosity_pa_s"],
            error_bounds["viscosity_pa_s"],
            samples,
        )
        flow_arr = np.random.normal(
            nominal_inputs["flow_rate_ml_h"],
            error_bounds["flow_rate_ml_h"],
            samples,
        )
        temp_arr = np.random.normal(
            nominal_inputs["temperature_celsius"],
            error_bounds["temperature_celsius"],
            samples,
        )

        # Enforce physical boundaries to prevent square root or exponential runtime overflows
        voltage_arr = np.maximum(voltage_arr, 1.0)
        viscosity_arr = np.maximum(viscosity_arr, 0.01)
        flow_arr = np.maximum(flow_arr, 0.01)

        # 2. Execute Level 4 and Level 1 physical core equations simultaneously
        fiber_diameters = self.k_factor * np.sqrt(
            (flow_arr * viscosity_arr) / voltage_arr
        )

        thermal_deviation = 0.5 * (temp_arr - self.lcst)
        # Numerical protection limit preventing floating-point overflow inside np.exp
        thermal_deviation = np.clip(thermal_deviation, -700.0, 700.0)
        swelling_ratios = 1.0 + (3.5 - 1.0) / (1.0 + np.exp(thermal_deviation))

        # 3. Map cross-correlations to create the final sensitivity matrix
        sensitivity_matrix = {
            "fiber_diameter_dependencies": {
                "voltage_kv_influence": self.compute_pearson_coefficient(
                    voltage_arr, fiber_diameters
                ),
                "viscosity_pa_s_influence": self.compute_pearson_coefficient(
                    viscosity_arr, fiber_diameters
                ),
                "flow_rate_ml_h_influence": self.compute_pearson_coefficient(
                    flow_arr, fiber_diameters
                ),
                "temperature_celsius_influence": self.compute_pearson_coefficient(
                    temp_arr, fiber_diameters
                ),
            },
            "swelling_ratio_dependencies": {
                "voltage_kv_influence": self.compute_pearson_coefficient(
                    voltage_arr, swelling_ratios
                ),
                "viscosity_pa_s_influence": self.compute_pearson_coefficient(
                    viscosity_arr, swelling_ratios
                ),
                "flow_rate_ml_h_influence": self.compute_pearson_coefficient(
                    flow_arr, swelling_ratios
                ),
                "temperature_celsius_influence": self.compute_pearson_coefficient(
                    temp_arr, swelling_ratios
                ),
            },
        }

        return {
            "status": "success",
            "samples_analyzed": samples,
            "sensitivity_indices": sensitivity_matrix,
            "raw_distribution_data_pures": {
                "fiber_diameter_nm": fiber_diameters.tolist(),
                "swelling_ratio": swelling_ratios.tolist(),
            },
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
        "temperature_celsius": 32.5,  # Critical LCST phase transition zone
    }

    # Observed hardware sensor standard deviation limits
    deviations = {
        "voltage_kv": 0.6,
        "viscosity_pa_s": 0.08,
        "flow_rate_ml_h": 0.03,
        "temperature_celsius": 1.0,
    }

    try:
        # Run the computation over 10,000 localized variations
        analysis_report = analyzer.analyze_system_sensitivity(
            setpoints, deviations, samples=10000
        )

        print("\n📈 [SENSITIVITY INDICES MATRIX RESULTS]:")
        fiber_deps = analysis_report["sensitivity_indices"][
            "fiber_diameter_dependencies"
        ]
        print(
            f"   ➤ Voltage influence on Fiber Size: {fiber_deps['voltage_kv_influence']:.4f} (Inverted directionality)"
        )
        print(
            f"   ➤ Viscosity influence on Fiber Size: {fiber_deps['viscosity_pa_s_influence']:.4f} (Direct relationship)"
        )

        swelling_deps = analysis_report["sensitivity_indices"][
            "swelling_ratio_dependencies"
        ]
        print(
            f"   ➤ Temperature influence on Hydrogel Swelling: {swelling_deps['temperature_celsius_influence']:.4f} (Strong negative shift)"
        )

    except ValueError as errors:
        print(f"❌ Execution failed on validation constraints: {errors}")