from typing import Any, Dict, Union
import numpy as np


class HydrogelEnergyEngine:
    """Energy Efficiency.

    Evaluates power overhead and Specific Energy Consumption (SEC) for manufacturing
    systems processing hydrogel solutions through electrostatic forces.
    """

    def __init__(self, solution_density_g_ml: float = 1.05):
        """Initializes the energetic engine with materials database baselines.

        Args:
            solution_density_g_ml (float, optional): Mass density of the target
                polymeric hydrogel solution in grams per milliliter (g/mL).
                Must be strictly positive. Defaults to 1.05.
        """
        if solution_density_g_ml <= 0:
            raise ValueError("Solution density must be strictly positive.")
        self.density = solution_density_g_ml

    def compute_power_consumption(
        self,
        voltage_kv: Union[float, np.ndarray],
        current_ua: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Calculates the net electrical power demanded by the high-voltage jet setup.

        Calculates instantaneous electric power using Joule's Law:
        $$P = V \\cdot I$$

        Where:
        - $P$: Power in Watts (W).
        - $V$: Applied potential field converted from kilovolts ($kV \\cdot 10^3$).
        - $I$: Current feedback converted from microamperes ($\\mu A \\cdot 10^{-6}$).

        Therefore, the balanced equation becomes:
        $$P = (V_{kV} \\cdot 10^3) \\cdot (I_{\\mu A} \\cdot 10^{-6}) = V_{kV} \\cdot I_{\\mu A} \\cdot 10^{-3}$$

        Args:
            voltage_kv (Union[float, np.ndarray]): Applied electrical potential in kV.
                Must contain non-negative elements.
            current_ua (Union[float, np.ndarray]): Collected corona or jet leakage
                current in microamperes. Must contain non-negative elements.

        Returns:
            Union[float, np.ndarray]: Net electrical power drawn by the system in Watts (W).

        Raises:
            ValueError: If any element in voltage_kv or current_ua is negative.
        """
        if np.any(voltage_kv < 0) or np.any(current_ua < 0):
            raise ValueError(
                "Electrical telemetry parameters cannot possess negative values."
            )

        power_watts = (voltage_kv * 1000.0) * (current_ua * 1e-6)
        return power_watts

    def compute_specific_energy_consumption(
        self,
        power_watts: Union[float, np.ndarray],
        flow_rate_ml_h: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Computes the Specific Energy Consumption (SEC).

        Represents the energy footprint required to process a unit mass of polymer:
        $$\\text{SEC} = \\frac{P_{\\text{kW}}}{\\dot{m}} = \\frac{P_{\\text{Watts}} / 1000.0}{Q_{\\text{mL/h}} \\cdot \\rho}$$

        Where:
        - $\\text{SEC}$: Specific Energy Consumption expressed in kilowatt-hours
          per gram (kWh/g).
        - $P_{\\text{kW}}$: High voltage power delivery in kilowatts.
        - $\\dot{m}$: Mass throughput stream in grams per hour ($g/h$).
        - $Q_{\\text{mL/h}}$: Volumetric injection flow rate array (`flow_rate_ml_h`).
        - $\\rho$: Hydrogel liquid formulation mass density (`density`).

        Args:
            power_watts (Union[float, np.ndarray]): Calculated power draw in Watts.
                Must contain non-negative values.
            flow_rate_ml_h (Union[float, np.ndarray]): Solution delivery pump speed.
                Must contain elements strictly greater than zero to avoid division by zero.

        Returns:
            Union[float, np.ndarray]: Energy footprint metrics resolved in kWh/g.

        Raises:
            ValueError: If power_watts contains negative values or flow_rate_ml_h
                contains values less than or equal to zero.
        """
        if np.any(power_watts < 0):
            raise ValueError("Power values cannot be negative.")
        if np.any(flow_rate_ml_h <= 0):
            raise ValueError(
                "Volumetric mass flow rate must be strictly positive (> 0 mL/h)."
            )

        # Mass feed rate calculation: flow_rate (mL/h) * density (g/mL) = g/h
        mass_flow_g_h = flow_rate_ml_h * self.density

        # SEC = (Power in kW) / (Mass flow in g/h) = kWh/g
        power_kw = power_watts / 1000.0
        sec_kwh_g = power_kw / mass_flow_g_h

        return sec_kwh_g

    def execute_energetic_pipeline(
        self, telemetry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Orchestrates Level 2 calculations handling vectorized matrix evaluations.

        Maps raw electric and fluidic telemetry diagnostics to continuous processing
        efficiency benchmarks.

        Args:
            telemetry (Dict[str, Any]): Dictionary containing real-time stream data.
                Required keys: "voltage_kv", "current_ua", and "flow_rate_ml_h".

        Returns:
            Dict[str, Any]: Consolidated metrics summary enclosing:
                - "power_watts_mean": Averaged electrical load.
                - "power_watts_std": Variance stability of electrical load.
                - "sec_kwh_g_mean": Calculated average mass efficiency footprint.
                - "sec_kwh_g_std": Deviations within production index runs.
                - "raw_energy_arrays": Serialized execution output vectors.

        Raises:
            ValueError: If structural telemetry values breach physics engine boundaries.
        """
        voltages = np.array(telemetry["voltage_kv"], dtype=float)
        currents = np.array(telemetry["current_ua"], dtype=float)
        flows = np.array(telemetry["flow_rate_ml_h"], dtype=float)

        # 1. Evaluate instant operational power steps
        calculated_power = self.compute_power_consumption(voltages, currents)

        # 2. Evaluate specific mass energy footprint requirements
        calculated_sec = self.compute_specific_energy_consumption(
            calculated_power, flows
        )

        return {
            "power_watts_mean": float(np.mean(calculated_power)),
            "power_watts_std": float(np.std(calculated_power)),
            "sec_kwh_g_mean": float(np.mean(calculated_sec)),
            "sec_kwh_g_std": float(np.std(calculated_sec)),
            "raw_energy_arrays": {
                "power_watts": (
                    calculated_power.tolist()
                    if calculated_power.ndim > 0
                    else float(calculated_power)
                ),
                "sec_kwh_g": (
                    calculated_sec.tolist()
                    if calculated_sec.ndim > 0
                    else float(calculated_sec)
                ),
            },
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
        "flow_rate_ml_h": [0.4, 0.45, 0.5, 0.5, 0.55],
    }

    try:
        # Process vectorized matrices through energy efficiency algorithms
        efficiency_summary = energy_core.execute_energetic_pipeline(
            test_telemetry_batch
        )

        print("\n⚡ [ENERGY METRICS GENERATED SUCCESSFULLY]:")
        print(
            f"   ➤ Average Power Draw: {efficiency_summary['power_watts_mean']:.3f} W"
        )
        print(
            f"   ➤ Specific Energy Consumption (SEC) Mean: {efficiency_summary['sec_kwh_g_mean']:.5f} kWh/g"
        )
        print(
            f"   ➤ SEC Standard Deviation: {efficiency_summary['sec_kwh_g_std']:.5f} kWh/g"
        )

    except ValueError as constraint_error:
        print(f"❌ Energetic pipeline execution halted: {constraint_error}")