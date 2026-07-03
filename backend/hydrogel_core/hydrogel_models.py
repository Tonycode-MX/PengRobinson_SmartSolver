from typing import Any, Dict, Union
import numpy as np


class HydrogelVectorizedEngine:
    """High-performance multi-scale mathematical core for hydrogel properties.

    Designed to process both scalar values and vectorized NumPy arrays simultaneously
    to enable high-throughput validation or synthetic dataset generation.
    """

    def __init__(
        self, lcst_celsius: float = 32.0, scaling_constant_k: float = 150.0
    ):
        """Initializes core thermodynamic and manufacturing macromolecular constants.

        Args:
            lcst_celsius (float, optional): Lower Critical Solution Temperature
                (LCST) in Celsius. Defaults to 32.0.
            scaling_constant_k (float, optional): Empirical proportionality
                factor for electrospun nanofiber diameter estimation. Defaults to 150.0.
        """
        self.lcst = lcst_celsius
        self.k_factor = scaling_constant_k

    def compute_fiber_diameter(
        self,
        voltage_kv: Union[float, np.ndarray],
        viscosity_pa_s: Union[float, np.ndarray],
        flow_rate_ml_h: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Computes nominal electrospun nanofiber diameters using scaling laws.

        The physical scaling relationship handles element-wise arrays or single scalars:
        $$d = k \\cdot \\sqrt{\\frac{Q \\cdot \\eta}{V}}$$

        Where:
        - $d$: Nanofiber diameter (`fiber_diameter_nm`).
        - $k$: Material empirical constant (`k_factor`).
        - $Q$: Solution feeding volumetric flow rate (`flow_rate_ml_h`).
        - $\\eta$: Polymeric dynamic viscosity (`viscosity_pa_s`).
        - $V$: Applied electrostatic potential field (`voltage_kv`).

        Args:
            voltage_kv (Union[float, np.ndarray]): Applied potential in kilovolts.
                Must contain strictly positive values to maintain field metrics.
            viscosity_pa_s (Union[float, np.ndarray]): Dynamic viscosity in Pa*s.
                Must contain values strictly greater than zero.
            flow_rate_ml_h (Union[float, np.ndarray]): Feeding rate in mL/h.
                Must contain values strictly greater than zero.

        Returns:
            Union[float, np.ndarray]: Calculated fiber diameter in nanometers (nm).
                Matches the shape and type constraints of the array inputs.

        Raises:
            ValueError: If any element inside the input parameters falls below or
                equals zero, breaking fluid dynamics definitions.
        """
        if np.any(voltage_kv <= 0):
            raise ValueError(
                "Voltage threshold inputs must contain strictly positive values (> 0 kV)."
            )
        if np.any(viscosity_pa_s <= 0):
            raise ValueError(
                "Viscosity metrics must contain strictly positive values (> 0 Pa*s)."
            )
        if np.any(flow_rate_ml_h <= 0):
            raise ValueError(
                "Flow rate arrays must contain strictly positive values (> 0 mL/h)."
            )

        # Mathematical scaling law executed at the hardware layer via NumPy
        operational_ratio = (flow_rate_ml_h * viscosity_pa_s) / voltage_kv
        fiber_diameter_nm = self.k_factor * np.sqrt(operational_ratio)

        return fiber_diameter_nm

    def compute_swelling_ratio(
        self,
        temperature_celsius: Union[float, np.ndarray],
        sensitivity_slope: float = 0.5,
    ) -> Union[float, np.ndarray]:
        """Models the sigmoidal thermal volume transition of the polymer network.

        Calculates continuous hydrogel volume phase transitions (HVPT) over arrays:
        $$S_r = S_{min} + \\frac{S_{max} - S_{min}}{1 + e^{\\alpha \\cdot (T - T_{LCST})}}$$

        Where:
        - $S_r$: Vector output of dimensional swelling ratios.
        - $S_{max}$, $S_{min}$: Maximum expanded (3.5) and fully collapsed (1.0) matrix constants.
        - $\\alpha$: Phase transformation rate coefficient (`sensitivity_slope`).
        - $T$: Target operating thermal states array (`temperature_celsius`).
        - $T_{LCST}$: Lower Critical Solution Temperature matrix switch (`lcst`).

        Args:
            temperature_celsius (Union[float, np.ndarray]): Operating matrix thermal status.
            sensitivity_slope (float, optional): Transition sharpness rate.
                Must be positive. Defaults to 0.5.

        Returns:
            Union[float, np.ndarray]: Swelling factors (dimensionless volume ratio).

        Raises:
            ValueError: If sensitivity_slope is less than or equal to zero.
        """
        if sensitivity_slope <= 0:
            raise ValueError("Sensitivity slope must be strictly positive.")

        max_swelling = 3.5  # Swollen matrix state threshold
        min_swelling = 1.0  # Collapsed matrix state threshold

        # NumPy vectorized exponential handles single floats or massive arrays instantly
        thermal_deviation = sensitivity_slope * (
            temperature_celsius - self.lcst
        )

        # Vectorized clipping prevents exponential array values from overflowing bit sizes
        thermal_deviation = np.clip(thermal_deviation, -700.0, 700.0)
        swelling_factor = min_swelling + (max_swelling - min_swelling) / (
            1.0 + np.exp(thermal_deviation)
        )

        return swelling_factor

    def execute_coupled_simulation(
        self, telemetry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Orchestrates multi-scale execution pipelines mapping processing inputs to physical states.

        Processes raw laboratory dictionaries containing combinations of scalar and
        iterable sequential structures smoothly through NumPy array formatting.

        Args:
            telemetry (Dict[str, Any]): Laboratory sensor input dictionary. Must contain keys:
                "voltage_kv", "viscosity_pa_s", "flow_rate_ml_h", and "temperature_celsius".

        Returns:
            Dict[str, Any]: Extracted macro statistics and clean data delivery structures:
                - "fiber_diameter_nm_mean": Calculated aggregate fiber mean.
                - "fiber_diameter_nm_std": Geometric fiber standard deviation.
                - "swelling_ratio_mean": Calculated hydrogel swelling state aggregate mean.
                - "swelling_ratio_std": Stochastic deviation of swelling response arrays.
                - "raw_arrays": Parsed lists optimized for transmission payload serialization.

        Raises:
            ValueError: If telemetry input fails underlying mathematical constraints.
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
                "fiber_diameter_nm": (
                    fiber_array.tolist()
                    if fiber_array.ndim > 0
                    else float(fiber_array)
                ),
                "swelling_ratio": (
                    swelling_array.tolist()
                    if swelling_array.ndim > 0
                    else float(swelling_array)
                ),
            },
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
        "temperature_celsius": [37.0, 36.8, 37.2, 37.0, 36.9],
    }

    try:
        # Process array payload across scaling laws
        array_results = engine.execute_coupled_simulation(
            experimental_grid_payload
        )

        print("\n📊 [VECTORIZED SIMULATION METRICS GENERATED]:")
        print(
            f"   ➤ Fiber Diameter Mean: {array_results['fiber_diameter_nm_mean']:.2f} nm"
        )
        print(
            f"   ➤ Fiber Diameter Std Dev: {array_results['fiber_diameter_nm_std']:.2f} nm"
        )
        print(
            f"   ➤ Network Swelling Mean: {array_results['swelling_ratio_mean']:.4f}"
        )

    except ValueError as validation_error:
        print(f"❌ Batch simulation aborted: {validation_error}")