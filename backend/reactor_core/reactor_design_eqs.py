import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, List, Union

class ReactorAITools:
    """
    A collection of static methods for chemical reactor design and simulation.
    
    Designed specifically to be used as tools by LLM Agents.
    Assumes first-order kinetics (A -> B) in the liquid phase (constant density).
    """

    @staticmethod
    def _rate_law(k: float, Ca0: float, X: float) -> float:
        """
        Calculates the rate of reaction (-rA) for a first-order liquid-phase reaction.

        Assumes constant volume/density, where the concentration of A is 
        defined as Ca = Ca0 * (1 - X).

        Args:
            k (float): Reaction rate constant (1/min).
            Ca0 (float): Initial concentration of reactant A (mol/L).
            X (float): Fractional conversion of reactant A (dimensionless, 0.0 to 1.0).

        Returns:
            float: The rate of disappearance of A (-rA) in mol/(L*min).
        """
        Ca = Ca0 * (1.0 - X)
        return k * Ca

    @staticmethod
    def simulate_batch(k: float, Ca0: float, t_final: float, X0: float = 0.0) -> Dict[str, Any]:
        """
        Simulates a Batch Reactor over a given time period using ordinary differential equations.

        Solves the mole balance for a batch reactor: dX/dt = -rA / Ca0.

        Args:
            k (float): Reaction rate constant (1/min). Must be strictly positive.
            Ca0 (float): Initial concentration of A (mol/L). Must be strictly positive.
            t_final (float): Total simulation time (min). Must be strictly positive.
            X0 (float, optional): Initial fractional conversion at t=0. Defaults to 0.0.

        Returns:
            Dict[str, Any]: A structured response containing:
                - "status" (str): "success" or "error".
                - "reactor_type" (str): "Batch".
                - "data" (dict): Contains "time_min" (List[float]), "conversion" (List[float]), 
                  and "final_conversion" (float).
                - "message" (str): Error description (only if status is "error").

        Raises:
            ValueError: If kinetic parameters or time are not strictly positive, 
                        or if X0 is outside the [0, 1) range.
        """
        try:
            if k <= 0 or Ca0 <= 0 or t_final <= 0:
                raise ValueError("Kinetic parameters and time must be strictly positive.")
            if not (0.0 <= X0 < 1.0):
                raise ValueError("Initial conversion X0 must be between 0 and 1.")

            def batch_model(t: float, X: List[float]) -> List[float]:
                ra = ReactorAITools._rate_law(k, Ca0, X[0])
                dX_dt = ra / Ca0
                return [dX_dt]

            sol = solve_ivp(batch_model, [0, t_final], [X0], dense_output=True)
            t_eval = np.linspace(0, t_final, 50)
            X_calc = sol.sol(t_eval)[0]

            return {
                "status": "success",
                "reactor_type": "Batch",
                "data": {
                    "time_min": t_eval.tolist(),
                    "conversion": X_calc.tolist(),
                    "final_conversion": float(X_calc[-1])
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def design_cstr(k: float, Ca0: float, v0: float, X_target: float) -> Dict[str, Any]:
        """
        Calculates the required volume for a Continuous Stirred-Tank Reactor (CSTR) 
        to achieve a specific target conversion.

        Uses the algebraic CSTR design equation: V = (Fa0 * X) / -rA.

        Args:
            k (float): Reaction rate constant (1/min). Must be strictly positive.
            Ca0 (float): Initial concentration of A (mol/L). Must be strictly positive.
            v0 (float): Volumetric flow rate (L/min). Must be strictly positive.
            X_target (float): Target fractional conversion (dimensionless, e.g., 0.8 for 80%).

        Returns:
            Dict[str, Any]: A structured response containing:
                - "status" (str): "success" or "error".
                - "reactor_type" (str): "CSTR".
                - "data" (dict): Contains "target_conversion" (float) and "required_volume_L" (float).
                - "message" (str): Error description (only if status is "error").

        Raises:
            ValueError: If kinetic parameters or flow rate are not strictly positive, 
                        or if X_target is not strictly between 0 and 1.
        """
        try:
            if k <= 0 or Ca0 <= 0 or v0 <= 0:
                raise ValueError("Kinetic parameters and volumetric flow must be strictly positive.")
            if not (0.0 < X_target < 1.0):
                raise ValueError("Target conversion must be strictly between 0 and 1.")

            Fa0 = Ca0 * v0
            ra = ReactorAITools._rate_law(k, Ca0, X_target)
            
            # CSTR Design Equation: V = (Fa0 * X) / -rA
            volume = (Fa0 * X_target) / ra

            return {
                "status": "success",
                "reactor_type": "CSTR",
                "data": {
                    "target_conversion": X_target,
                    "required_volume_L": float(volume)
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def simulate_pfr(k: float, Ca0: float, v0: float, V_final: float, X0: float = 0.0) -> Dict[str, Any]:
        """
        Simulates a Plug Flow Reactor (PFR) along its volume to calculate the conversion profile.

        Solves the mole balance for a PFR: dX/dV = -rA / Fa0.

        Args:
            k (float): Reaction rate constant (1/min). Must be strictly positive.
            Ca0 (float): Initial concentration of A (mol/L). Must be strictly positive.
            v0 (float): Volumetric flow rate (L/min). Must be strictly positive.
            V_final (float): Total volume of the reactor (L). Must be strictly positive.
            X0 (float, optional): Initial fractional conversion at V=0. Defaults to 0.0.

        Returns:
            Dict[str, Any]: A structured response containing:
                - "status" (str): "success" or "error".
                - "reactor_type" (str): "PFR".
                - "data" (dict): Contains "volume_L" (List[float]), "conversion" (List[float]), 
                  and "final_conversion" (float).
                - "message" (str): Error description (only if status is "error").

        Raises:
            ValueError: If parameters, flow, or volume are not strictly positive, 
                        or if X0 is outside the [0, 1) range.
        """
        try:
            if k <= 0 or Ca0 <= 0 or v0 <= 0 or V_final <= 0:
                raise ValueError("Parameters, flow, and volume must be strictly positive.")
            if not (0.0 <= X0 < 1.0):
                raise ValueError("Initial conversion X0 must be between 0 and 1.")

            Fa0 = Ca0 * v0

            def pfr_model(V: float, X: List[float]) -> List[float]:
                ra = ReactorAITools._rate_law(k, Ca0, X[0])
                dX_dV = ra / Fa0
                return [dX_dV]

            sol = solve_ivp(pfr_model, [0, V_final], [X0], dense_output=True)
            V_eval = np.linspace(0, V_final, 50)
            X_calc = sol.sol(V_eval)[0]

            return {
                "status": "success",
                "reactor_type": "PFR",
                "data": {
                    "volume_L": V_eval.tolist(),
                    "conversion": X_calc.tolist(),
                    "final_conversion": float(X_calc[-1])
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

# ==========================================
# 🤖 EXAMPLE: HOW AN AGENT WOULD CALL THIS
# ==========================================
if __name__ == "__main__":
    # Agent decides to calculate a CSTR volume for 85% conversion
    agent_response = ReactorAITools.design_cstr(
        k=0.15, 
        Ca0=1.2, 
        v0=5.0, 
        X_target=0.85
    )
    
    print("Agent Execution Result:")
    import json
    print(json.dumps(agent_response, indent=2))
