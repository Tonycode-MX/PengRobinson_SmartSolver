import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool, tool

# Programmatic system path alignment to prevent modular import routing failure cascades
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(CURRENT_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from agents.llm_setup import llm
from utils.conversions import conversion_tools_list
from grapher.hydrogel_plots import HydrogelGrapher3D

# Initialize a persistent singleton instance of the local 3D Plotly serialization engine
graph_engine = HydrogelGrapher3D()

# Persistent operational laboratory state baseline memory registry (Simulated RAM Database)
LAB_NOMINAL_MEMORY: Dict[str, float] = {
    "voltage_kv": 20.0,
    "viscosity_pa_s": 1.2,
    "flow_rate_ml_h": 0.5,
    "temperature_celsius": 32.5,
    "current_ua": 1.2
}

# Physical laboratory safe engineering boundaries restricting continuous process scaling
SAFETY_BOUNDS: Dict[str, tuple] = {
    "voltage_kv": (1.0, 35.0),
    "viscosity_pa_s": (0.05, 15.0),
    "flow_rate_ml_h": (0.01, 5.0),
    "temperature_celsius": (10.0, 60.0),
    "current_ua": (0.01, 10.0)
}

# Empirical kinetic parameters for hydrogel pore thermo-response (PNIPAM-based LCST baseline)
LCST_TEMPERATURE = 32.0         # Lower Critical Solution Temperature in Celsius
MAX_PORE_SHRINKAGE_RATE = 0.85   # Maximum volumetric contraction ratio at hyperthermia

# =====================================================================
# AI TOOLS FOR HYDROGEL SPECIALIST AGENT
# =====================================================================

@tool
def tool_read_current_laboratory_state() -> str:
    """
    Queries and reads the persistent operational telemetry registry in the system memory.
    Use this tool prior to calculating any mathematical deltas or verifying active manufacturing baselines.
    
    Returns:
        str: A clean data string formatting the active parameters of the laboratory manufacturing cell.
    """
    return (
        "System Observation: Current baseline active manufacturing configuration readouts:\n"
        f"• Voltage: {LAB_NOMINAL_MEMORY['voltage_kv']:.2f} kV\n"
        f"• Polymer Viscosity: {LAB_NOMINAL_MEMORY['viscosity_pa_s']:.2f} Pa·s\n"
        f"• Syringe Flow Rate: {LAB_NOMINAL_MEMORY['flow_rate_ml_h']:.2f} mL/h\n"
        f"• Ambient Chamber Temperature: {LAB_NOMINAL_MEMORY['temperature_celsius']:.2f} °C\n"
        f"• Jet Current: {LAB_NOMINAL_MEMORY['current_ua']:.2f} uA"
    )

@tool
def tool_modify_and_simulate_hydrogel(
    parameter_key: str,
    modification_type: str,
    value: float,
    experimental_batch_nm: Optional[List[float]] = None
) -> str:
    """
    Fase 1: Transforms continuous fabrication values, enforces physical safety constraints, 
    and executes a high-fidelity stochastic Monte Carlo distribution loop (10,000 samples)
    generating a 3D response surface mapping manufacturing parameter variance.
    
    Args:
        parameter_key (str): Target parameter. Must be one of: 'voltage_kv', 'viscosity_pa_s', 'flow_rate_ml_h', 'temperature_celsius', 'current_ua'.
        modification_type (str): Arithmetic profile to compute. Must be one of: 'absolute_set', 'unit_increase', 'unit_decrease', 'percentage_increase', 'percentage_decrease'.
        value (float): The numeric magnitude of delta or absolute target.
        experimental_batch_nm (List[float], optional): Optional real discrete microscopic measurements to superimpose on the Plotly layout.
        
    Returns:
        str: A data-dense summary reporting settled state values, safety actions, and distribution statistics.
    """
    if parameter_key not in LAB_NOMINAL_MEMORY:
        return f"System Error: The requested target operational key '{parameter_key}' does not exist inside state registries."

    current_nominal = LAB_NOMINAL_MEMORY[parameter_key]
    target_value = current_nominal

    # Execute dynamic math assignment mappings
    if modification_type == "absolute_set":
        target_value = value
    elif modification_type == "unit_increase":
        target_value = current_nominal + value
    elif modification_type == "unit_decrease":
        target_value = current_nominal - value
    elif modification_type == "percentage_increase":
        target_value = current_nominal * (1.0 + (value / 100.0))
    elif modification_type == "percentage_decrease":
        target_value = current_nominal * (1.0 - (value / 100.0))
    else:
        return f"System Error: Unrecognized arithmetic manipulation rule set profile: '{modification_type}'."

    # Enforce rigid physical security clamp triggers (Clamping safeguards)
    min_bound, max_bound = SAFETY_BOUNDS[parameter_key]
    safety_triggered = False
    safety_message = ""

    if target_value > max_bound:
        clamped_value = max_bound
        safety_triggered = True
        safety_message = f"CRITICAL SAFETY BOUNDARY CLAMPED: Target value ({target_value:.2f}) exceeded max safe limits ({max_bound:.2f})."
    elif target_value < min_bound:
        clamped_value = min_bound
        safety_triggered = True
        safety_message = f"CRITICAL SAFETY BOUNDARY CLAMPED: Target value ({target_value:.2f}) dropped below min safe limits ({min_bound:.2f})."
    else:
        clamped_value = target_value

    # Commit state mutation securely back into the active process RAM window
    LAB_NOMINAL_MEMORY[parameter_key] = round(clamped_value, 4)

    # Launch the high-performance stochastic Monte Carlo simulation matrix (10,000 runs)
    samples = 10000
    rng = np.random.default_rng()

    # Model physical environmental sensor noise distributions (Gaussian models)
    voltages = rng.normal(LAB_NOMINAL_MEMORY["voltage_kv"], 0.4, samples)
    voltages = np.maximum(voltages, 1.0)  # Asymptotic geometric floor protection to avoid division by zero
    viscosities = rng.normal(LAB_NOMINAL_MEMORY["viscosity_pa_s"], 0.05, samples)
    flows = rng.normal(LAB_NOMINAL_MEMORY["flow_rate_ml_h"], 0.01, samples)

    # Semi-empirical electrospinning physical scaling laws
    k_factor = 1267.83
    simulated_diameters = (k_factor * np.sqrt((flows * viscosities) / voltages)).tolist()

    # Assign default physical validation datasets if none passed down the wire
    fallback_laboratory_data = [215.4, 222.1, 219.8, 228.3, 211.0, 220.5]
    active_lab_array = experimental_batch_nm if experimental_batch_nm else fallback_laboratory_data

    # Compile the resulting matrix arrays into a single raw Plotly 3D JSON document
    serialized_chart_json = graph_engine.generate_surface_distribution_3d_json(
        simulated_diameters=simulated_diameters,
        voltages_kv=voltages.tolist(),
        experimental_diameters=active_lab_array
    )

    # Cache the rendering context payload inside environmental communication variables
    os.environ["LATEST_PLOT_JSON"] = serialized_chart_json

    # Build the non-bleeding observation statement string for agent extraction
    output_string = (
        "System Observation: Hydrogel manufacturing processing simulation loop completed successfully.\n"
        f"• Modified Parameter Reference: '{parameter_key}'\n"
        f"• Arithmetic Vector Profile: '{modification_type}' (Value Shift: {value})\n"
        f"• Settled Registry Configuration: {LAB_NOMINAL_MEMORY[parameter_key]} (Previous state: {current_nominal})\n"
        f"• Continuous Safety Intervention Active: {safety_triggered}\n"
    )
    if safety_triggered:
        output_string += f"• Intercept Context Warning: {safety_message}\n"

    output_string += (
        "\n--- STOCHASTIC MANUFACTURE DISTRIBUTIONS ANALYTICS ---\n"
        f"• Predicted Mean Fiber Size: {np.mean(simulated_diameters):.2f} nm\n"
        f"• Predicted Geometric Standard Deviation: {np.std(simulated_diameters):.2f} nm\n"
        "• Plotly Multi-Layer 3D Surface Status: Fabrication payload package compiled successfully."
    )
    return output_string

@tool
def tool_simulate_thermal_pore_shrinkage_release(
    target_body_temperature: float,
    total_simulation_hours: float
) -> str:
    """
    Fase 2: Simulates the physiological hydrogel pore shrinkage kinetics and calculates 
    the cumulative drug delivery release curves based on human body temperature and diffusion.
    
    Args:
        target_body_temperature (float): The host physiological temperature in Celsius (e.g., 37.0).
        total_simulation_hours (float): The continuous evaluation time window in hours.
    """
    # Enforce physiological safety clamping on human body inputs
    target_body_temperature = max(10.0, min(target_body_temperature, 60.0))
    total_simulation_hours = max(0.1, min(total_simulation_hours, 168.0)) # Max 1 week window

    # Define grid spacing arrays for high-fidelity 3D mesh surface representation
    temps = np.linspace(25.0, 42.0, 30)  # Thermal sweep across hypothermia to hyperthermia bounds
    times = np.linspace(0.0, total_simulation_hours, 40)
    
    times_grid, temps_grid = np.meshgrid(times, temps)
    
    # Mathematical thermodynamic model mapping volumetric pore shrinking via thermal activation (Sigmoidal LCST Transition)
    shrinkage_factor = 1.0 - (MAX_PORE_SHRINKAGE_RATE / (1.0 + np.exp(-(temps_grid - LCST_TEMPERATURE) / 1.5)))
    
    # Mass transfer diffusion equations (Modified structural Korsmeyer-Peppas matrix model)
    kinetic_exponent = 0.5 * (2.0 - shrinkage_factor)  # Structural anomalous macromolecular transport vector
    k_constant = 0.15 * (1.0 / shrinkage_factor)
    
    fraction_released = k_constant * np.power(times_grid, kinetic_exponent)
    fraction_released = np.clip(fraction_released, 0.0, 1.0)  # Bound tightly by law of conservation of mass

    # Dispatch data layers directly to the modern 3D Plotly serialization engine interface
    serialized_json_3d = graph_engine.generate_drug_release_surface_3d_json(
        times_hours=times_grid.flatten().tolist(),
        temperatures_celsius=temps_grid.flatten().tolist(),
        fraction_released=fraction_released.flatten().tolist()
    )
    
    # Store the 3D grid context payload inside environmental network variables
    os.environ["LATEST_PLOT_JSON"] = serialized_json_3d
    
    # Calculate specific point metrics for the user's explicit target request parameters
    specific_shrinkage = 1.0 - (MAX_PORE_SHRINKAGE_RATE / (1.0 + np.exp(-(target_body_temperature - LCST_TEMPERATURE) / 1.5)))
    specific_exponent = 0.5 * (2.0 - specific_shrinkage)
    final_release = np.clip(0.15 * (1.0 / specific_shrinkage) * np.power(total_simulation_hours, specific_exponent), 0.0, 1.0)

    return (
        "System Observation: Physiological thermal pore contraction and drug mass transfer diffusion executed.\n"
        f"• Evaluated Target Body Temperature: {target_body_temperature:.1f} °C\n"
        f"• Volumetric Matrix Pore Contraction: {(1.0 - specific_shrinkage) * 100:.2f}% reduction in pore volume\n"
        f"• Mass Transport Release Mechanism: Anomalous non-Fickian diffusion (n = {specific_exponent:.3f})\n"
        f"• Predicted Cumulative Drug Released at {total_simulation_hours} Hours: {final_release * 100:.2f}% of total encapsulated dosage.\n"
        "• Plotly 3D Surface Matrix Status: Delivery kinetics response surface compiled successfully."
    )

# Combine structural conversion utility lists with the unified dual-phase hydrogel tools
hydrogel_tools = conversion_tools_list + [
    tool_read_current_laboratory_state,
    tool_modify_and_simulate_hydrogel,
    tool_simulate_thermal_pore_shrinkage_release
]

# =====================================================================
# AGENT SETUP
# =====================================================================
hydrogel_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Hydrogel Synthesis and Electrospinning Specialist Agent.
    You manage laboratory telemetry configurations and run high-fidelity Monte Carlo size distribution simulations.
    
    Always execute operational steps in this STRICT sequential order:
    1. If the user statement invokes any relative tracking delta words (e.g., 'increase', 'subir', 'reduce', 'bajar', 'modify') or queries status, call 'tool_read_current_laboratory_state' first to look up active baselines.
    2. Route multi-language human vocabularies to explicit data dictionary parameters safely:
       - 'voltaje', 'voltage', 'kv' -> 'voltage_kv'
       - 'flujo', 'flow', 'ml/h', 'tasa', 'bombeo' -> 'flow_rate_ml_h'
       - 'viscosidad', 'viscosity' -> 'viscosity_pa_s'
       - 'temperatura', 'temperature', 'celsius' -> 'temperature_celsius'
       - 'corriente', 'current', 'ua' -> 'current_ua'
    3. If the request relates to the manufacturing phase (voltages, flow rates, viscosities), determine the calculation profile and call 'tool_modify_and_simulate_hydrogel'.
    4. If the request relates to clinical application (body temperature, drug delivery, pore shrinkage, time kinetics), extract variables and call 'tool_simulate_thermal_pore_shrinkage_release'.
    5. Provide the exact text summary results back to the user clearly using Markdown notation (bullet items and bold highlights).
    6. Conclude by writing a brief 1-2 sentence high-level scientific deduction explaining how this parameter shift physically controls taylor cone equilibrium or macromolecular polymer network aggregation, or how temperature triggers hydrogel LCST phase transition deswelling and anomalous drug mass transport."""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

hydrogel_agent = create_tool_calling_agent(llm, hydrogel_tools, hydrogel_prompt)
hydrogel_executor = AgentExecutor(agent=hydrogel_agent, tools=hydrogel_tools, verbose=False)

@tool(return_direct=True)
def hydrogel_specialist_tool(query: str) -> str:
    """
    Use this tool ONLY when the user's task or query involves hydrogel synthesis, electrospinning parameters, 
    macromolecular polymer fiber diameters, syringe pump flow rates, physiological drug delivery kinetics, 
    pore shrinkage mechanisms, time-release profiles, or stochastic Monte Carlo size distributions.
    """
    return hydrogel_executor.invoke({"input": query})["output"]