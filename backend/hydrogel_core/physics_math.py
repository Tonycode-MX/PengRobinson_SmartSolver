import math
from typing import Any, Dict, Tuple
import numpy as np


class HydrogelPhysicsMath:
    """Pure mathematical and physical core engine for hydrogel simulation.

    Handles multi-scale modeling using vectorized NumPy operations, linking
    electrospinning manufacturing parameters with thermodynamic phase transitions.
    """

    def __init__(
        self, lcst_celsius: float = 32.0, empirical_constant_k: float = 150.0
    ):
        """Initializes the thermodynamic and manufacturing constants.

        Args:
            lcst_celsius (float, optional): Lower Critical Solution Temperature
                (LCST) in Celsius. Defines the thermal boundary where the polymer
                undergoes a phase transition from hydrophilic to hydrophobic.
                Defaults to 32.0.
            empirical_constant_k (float, optional): Material and system-dependent
                scaling factor for electrospinning fiber diameter estimation.
                Defaults to 150.0.
        """
        self.lcst = lcst_celsius
        self.k_factor = empirical_constant_k

    def calculate_electrospinning_diameter(
        self, voltage_kv: float, viscosity_pa_s: float, flow_rate_ml_h: float
    ) -> float:
        """Computes the nominal nanofiber diameter using fluid dynamics scaling laws.

        The physical behavior is modeled via an analytical scaling law relation:
        $$d = k \cdot \\sqrt{\\frac{Q \\cdot \\eta}{V}}$$

        Where:
        - $d$ is the fiber diameter (nm)
        - $k$ is the empirical constant (`k_factor`)
        - $Q$ is the solution volumetric flow rate (`flow_rate_ml_h`)
        - $\\eta$ is the dynamic viscosity (`viscosity_pa_s`)
        - $V$ is the applied electrostatic potential (`voltage_kv`)

        This law balances the viscous forces and fluid feeding rate against the
        electrostatic drawing forces that stretch the Taylor cone jet.

        Args:
            voltage_kv (float): Applied electrical potential in kilovolts.
                Must be strictly positive to establish the electric field.
            viscosity_pa_s (float): Dynamic viscosity of the polymer solution.
                Must be strictly positive to represent a physical fluid.
            flow_rate_ml_h (float): Solution feeding rate from the pump.
                Must be strictly positive to sustain the jet.

        Returns:
            float: Nanofiber diameter in nanometers (nm).

        Raises:
            ValueError: If voltage_kv, viscosity_pa_s, or flow_rate_ml_h are
                less than or equal to zero, as they violate physical boundaries.
        """
        if voltage_kv <= 0:
            raise ValueError(
                "Voltage must be strictly positive to establish an electrostatic jet."
            )
        if viscosity_pa_s <= 0:
            raise ValueError("Viscosity must be strictly positive.")
        if flow_rate_ml_h <= 0:
            raise ValueError("Flow rate must be strictly positive.")

        # Analytical scaling law equation application
        dimensionless_ratio = (flow_rate_ml_h * viscosity_pa_s) / voltage_kv
        fiber_diameter_nm = self.k_factor * np.sqrt(dimensionless_ratio)

        return float(fiber_diameter_nm)

    def calculate_thermal_swelling_factor(
        self, temperature_celsius: float, alpha_sensitivity: float = 0.5
    ) -> float:
        """Models the volumetric thermal collapse of the hydrogel network.

        Uses a continuous sigmoidal transition profile around the LCST boundary
        to map the Hydrogel Volume Phase Transition (HVPT):
        $$S_f = S_{min} + \\frac{S_{max} - S_{min}}{1 + e^{\\alpha \cdot (T - T_{LCST})}}$$

        Where:
        - $S_f$ is the resulting dimensionless swelling factor.
        - $S_{max}$ is the fully hydrated/swollen state volume ratio (3.5).
        - $S_{min}$ is the collapsed state volume ratio (1.0).
        - $\\alpha$ is the steepness parameter (`alpha_sensitivity`).
        - $T$ is the current operating temperature (`temperature_celsius`).
        - $T_{LCST}$ is the thermal phase boundary (`lcst`).

        Physical intuition: When $T < T_{LCST}$, hydrogen bonds with water dominate,
        causing matrix expansion ($S_f \\to S_{max}$). When $T > T_{LCST}$,
        hydrophobic interactions dominate, expelling water and collapsing the network
        ($S_f \\to S_{min}$).

        Args:
            temperature_celsius (float): Operating thermal state of the matrix.
            alpha_sensitivity (float, optional): Steepness coefficient controlling
                how abrupt the phase transition occurs around the LCST boundary.
                Must be strictly positive. Defaults to 0.5.

        Returns:
            float: Dimensionless swelling ratio (V_swollen / V_collapsed).

        Raises:
            ValueError: If alpha_sensitivity is less than or equal to zero.
        """
        if alpha_sensitivity <= 0:
            raise ValueError("Alpha sensitivity coefficient must be positive.")

        swelling_max = 3.5  # Fully hydrated state value
        swelling_min = 1.0  # Collapsed network state value

        # Logistic sigmoidal distribution mapping the hydrogel volume phase transition (HVPT)
        exponent = alpha_sensitivity * (temperature_celsius - self.lcst)

        # Prevent overflow errors with extremely large positive exponents (extreme temperatures)
        if exponent > 700:
            return float(swelling_min)

        swelling_factor = swelling_min + (swelling_max - swelling_min) / (
            1.0 + np.exp(exponent)
        )

        return float(swelling_factor)

    def evaluate_coupled_system(
        self,
        voltage_kv: float,
        viscosity_pa_s: float,
        flow_rate_ml_h: float,
        temperature_celsius: float,
    ) -> Dict[str, float]:
        """Orchestrates mathematical execution coupling structural parameters with thermodynamics.

        Runs both the manufacturing execution model and the thermodynamic matrix
        resolution sequentially to describe the behavior of the system.

        Args:
            voltage_kv (float): Electrospinning voltage in kV.
            viscosity_pa_s (float): Solution dynamic viscosity in Pa*s.
            flow_rate_ml_h (float): Volumetric injection flow rate in mL/h.
            temperature_celsius (float): Environmental/physiological temperature.

        Returns:
            Dict[str, float]: Combined physical properties containing:
                - "fiber_diameter_nm": Calculated fiber diameter rounded to 2 decimals.
                - "swelling_ratio_dimensionless": Swelling ratio rounded to 4 decimals.

        Raises:
            ValueError: If any sub-function parameter validation fails.
        """
        # Execute manufacturing level matrix resolution
        fiber_diameter = self.calculate_electrospinning_diameter(
            voltage_kv, viscosity_pa_s, flow_rate_ml_h
        )

        # Execute thermodynamic level matrix resolution
        swelling_ratio = self.calculate_thermal_swelling_factor(
            temperature_celsius
        )

        return {
            "fiber_diameter_nm": round(fiber_diameter, 2),
            "swelling_ratio_dimensionless": round(swelling_ratio, 4),
        }


# =====================================================================
# STANDALONE MATHEMATICAL TESTING BENCH (SANITY CHECK)
# =====================================================================
if __name__ == "__main__":
    print("Testing pure mathematical functions under isolated parameters...")

    # Instantiate the physics engine core
    engine = HydrogelPhysicsMath()

    try:
        # Scenario A: Baseline standard execution validation (Physiological State)
        results = engine.evaluate_coupled_system(
            voltage_kv=20.0,
            viscosity_pa_s=1.2,
            flow_rate_ml_h=0.5,
            temperature_celsius=37.0,  # Above LCST, matrix should be collapsed
        )
        print(f"   ➤ Physiological State Results (37°C): {results}")

        # Scenario B: Ambient temperature execution validation (Room Temperature)
        results_cold = engine.evaluate_coupled_system(
            voltage_kv=20.0,
            viscosity_pa_s=1.2,
            flow_rate_ml_h=0.5,
            temperature_celsius=22.0,  # Below LCST, matrix should be swollen
        )
        print(f"   ➤ Room Temperature Results (22°C): {results_cold}")

    except ValueError as error:
        print(f"   ❌ Validation failure triggered: {error}")