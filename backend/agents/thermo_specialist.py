import numpy as np
from typing import List
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool, tool

from agents.llm_setup import llm
from utils.conversions import conversion_tools_list

from thermo_core.processes import (
    simulate_isobaric_process,
    simulate_isothermal_process,
    simulate_adiabatic_process,
    simulate_isochoric_process
)
from grapher.processes_plot import plot_thermodynamic_process

# =====================================================================
# AI TOOLS FOR THERMODYNAMIC SPECIALIST AGENT
# =====================================================================
@tool
def tool_isothermal_process(
    temperature: float, 
    p_start: float, 
    p_end: float, 
    fluids: List[str], 
    fractions: List[float]
) -> str:
    """
    Simulates an isothermal thermodynamic process and automatically generates a P-V interactive plot.
    
    This tool computes the Peng-Robinson Equation of State (EOS) trajectory for a real gas mixture 
    at a constant temperature across a pressure gradient. It tracks compressibility, molar volume, 
    and shifts in residual enthalpy, entropy, and Gibbs Free Energy. It renders an HTML dashboard 
    and returns the key physical property shifts to the agent.
    
    Args:
        temperature (float): The constant absolute temperature of the system in Kelvin (K).
        p_start (float): The initial absolute pressure in Pascals (Pa).
        p_end (float): The final absolute pressure in Pascals (Pa).
        fluids (List[str]): List of valid fluid names (e.g., ['Methane', 'Ethane']).
        fractions (List[float]): List of molar fractions corresponding to the fluids (must sum to 1.0).
        
    Returns:
        str: A formatted, data-only summary of the initial and final thermodynamic states.
    """
    # Initialize a zero-matrix for binary interaction parameters (k_ij)
    kij = np.zeros((len(fluids), len(fluids)))
    
    # Execute the backend mathematical simulation
    data = simulate_isothermal_process(temperature, p_start, p_end, fluids, fractions, kij)
    
    # Dispatch the resulting trajectory to the local Plotly renderer
    plot_thermodynamic_process(data, fluids, fractions) 
    
    # Return a clean string to the LLM agent to prevent prompt bleeding
    return (
        "System Observation: The isothermal simulation was executed and the HTML graph was rendered successfully. "
        "Please present the following key results to the user in a natural, professional manner:\n\n"
        f"• Initial Z-factor: {data['z_factors'][0]:.4f} | Final Z-factor: {data['z_factors'][-1]:.4f}\n"
        f"• Initial Molar Volume: {data['volumes_m3_mol'][0]:.6e} m³/mol | Final Molar Volume: {data['volumes_m3_mol'][-1]:.6e} m³/mol\n"
        f"• Change in Residual Enthalpy (ΔH): {data['delta_H_J_mol'][-1]:.2f} J/mol\n"
        f"• Change in Residual Entropy (ΔS): {data['delta_S_J_mol_K'][-1]:.4f} J/(mol·K)\n"
        f"• Change in Residual Gibbs Free Energy (ΔG): {data['delta_G_J_mol'][-1]:.2f} J/mol"
    )

@tool
def tool_isobaric_process(
    pressure: float, 
    t_start: float, 
    t_end: float, 
    fluids: List[str], 
    fractions: List[float]
) -> str:
    """
    Simulates an isobaric (constant pressure) thermodynamic process and automatically generates an interactive plot.
    
    This tool computes the Peng-Robinson Equation of State (EOS) heating or cooling trajectory 
    for a real gas mixture at a constant pressure across a temperature gradient. It renders 
    an HTML dashboard and returns the key physical property shifts to the agent.
    
    Args:
        pressure (float): The constant absolute pressure of the system in Pascals (Pa).
        t_start (float): The initial absolute temperature in Kelvin (K).
        t_end (float): The final absolute temperature in Kelvin (K).
        fluids (List[str]): List of valid fluid names (e.g., ['Methane', 'Ethane']).
        fractions (List[float]): List of molar fractions corresponding to the fluids (must sum to 1.0).
        
    Returns:
        str: A formatted, data-only summary of the initial and final thermodynamic states.
    """
    # Initialize a zero-matrix for binary interaction parameters (k_ij)
    kij = np.zeros((len(fluids), len(fluids)))
    
    # Execute the backend mathematical simulation
    data = simulate_isobaric_process(pressure, t_start, t_end, fluids, fractions, kij)
    
    # Dispatch the resulting trajectory to the local Plotly renderer
    plot_thermodynamic_process(data, fluids, fractions) 
    
    # Return a clean string to the LLM agent to prevent prompt bleeding
    return (
        "System Observation: The isobaric simulation was executed and the HTML graph was rendered successfully. "
        "Please present the following key results to the user in a natural, professional manner:\n\n"
        f"• Initial Z-factor: {data['z_factors'][0]:.4f} | Final Z-factor: {data['z_factors'][-1]:.4f}\n"
        f"• Initial Molar Volume: {data['volumes_m3_mol'][0]:.6e} m³/mol | Final Molar Volume: {data['volumes_m3_mol'][-1]:.6e} m³/mol\n"
        f"• Change in Residual Enthalpy (ΔH): {data['delta_H_J_mol'][-1]:.2f} J/mol\n"
        f"• Change in Residual Entropy (ΔS): {data['delta_S_J_mol_K'][-1]:.4f} J/(mol·K)\n"
        f"• Change in Residual Gibbs Free Energy (ΔG): {data['delta_G_J_mol'][-1]:.2f} J/mol"
    )

@tool
def tool_adiabatic_process(
    p_start: float, 
    p_end: float, 
    t_start: float, 
    fluids: List[str], 
    fractions: List[float]
) -> str:
    """
    Simulates a reversible adiabatic (isentropic) thermodynamic process and automatically generates an interactive plot.
    
    This tool computes the Peng-Robinson Equation of State (EOS) trajectory for a real gas mixture 
    undergoing adiabatic compression or expansion. It numerically finds the final temperature 
    where the entropy change is zero (ΔS = 0), calculates the steady-flow work, renders an HTML 
    dashboard, and returns the key physical property shifts to the agent.
    
    Args:
        p_start (float): The initial absolute pressure of the system in Pascals (Pa).
        p_end (float): The final absolute pressure of the system in Pascals (Pa).
        t_start (float): The initial absolute temperature in Kelvin (K).
        fluids (List[str]): List of valid fluid names (e.g., ['Methane', 'Ethane']).
        fractions (List[float]): List of molar fractions corresponding to the fluids (must sum to 1.0).
        
    Returns:
        str: A formatted, data-only summary of the initial and final thermodynamic states, including work and energy shifts.
    """
    # Initialize a zero-matrix for binary interaction parameters (k_ij)
    kij = np.zeros((len(fluids), len(fluids)))
    
    # Execute the backend mathematical simulation
    data = simulate_adiabatic_process(p_start, p_end, t_start, fluids, fractions, kij)
    
    # Dispatch the resulting trajectory to the local Plotly renderer
    plot_thermodynamic_process(data, fluids, fractions) 
    
    # Return a clean string to the LLM agent to prevent prompt bleeding
    return (
        "System Observation: The adiabatic (isentropic) simulation was executed and the HTML graph was rendered successfully. "
        "Please present the following key results to the user in a natural, professional manner:\n\n"
        f"• Initial Temperature: {data['temperatures_K'][0]:.2f} K | Final Isentropic Temperature: {data['temperatures_K'][-1]:.2f} K\n"
        f"• Initial Z-factor: {data['z_factors'][0]:.4f} | Final Z-factor: {data['z_factors'][-1]:.4f}\n"
        f"• Initial Molar Volume: {data['volumes_m3_mol'][0]:.6e} m³/mol | Final Molar Volume: {data['volumes_m3_mol'][-1]:.6e} m³/mol\n"
        f"• Steady-Flow Work Produced/Consumed: {data['work_produced_J_mol'][-1]:.2f} J/mol\n"
        f"• Change in Residual Internal Energy (ΔU): {data['delta_U_J_mol'][-1]:.2f} J/mol\n"
        f"• Change in Residual Entropy (ΔS): {data['delta_S_J_mol_K'][-1]:.2e} J/(mol·K) (Verification of isentropic state)"
    )

@tool
def tool_isochoric_process(
    v_m3_mol: float, 
    t_start: float, 
    t_end: float, 
    fluids: List[str], 
    fractions: List[float]
) -> str:
    """
    Simulates an isochoric (constant molar volume) thermodynamic process and automatically generates an interactive plot.
    
    This tool computes the Peng-Robinson Equation of State (EOS) heating or cooling trajectory 
    for a real gas mixture confined to a strict constant volume. It tracks the evolution of pressure, 
    compressibility, and shifts in residual internal energy (ΔU), Helmholtz Free Energy (ΔA), 
    and entropy. It renders an HTML dashboard and returns the key physical shifts to the agent.
    
    Args:
        v_m3_mol (float): The constant molar volume of the system in cubic meters per mole (m³/mol).
        t_start (float): The initial absolute temperature in Kelvin (K).
        t_end (float): The final absolute temperature in Kelvin (K).
        fluids (List[str]): List of valid fluid names (e.g., ['Methane', 'Ethane']).
        fractions (List[float]): List of molar fractions corresponding to the fluids (must sum to 1.0).
        
    Returns:
        str: A formatted, data-only summary of the initial and final thermodynamic states.
    """
    # Initialize a zero-matrix for binary interaction parameters (k_ij)
    kij = np.zeros((len(fluids), len(fluids)))
    
    # Execute the backend mathematical simulation
    data = simulate_isochoric_process(v_m3_mol, t_start, t_end, fluids, fractions, kij)
    
    # Dispatch the resulting trajectory to the local Plotly renderer
    plot_thermodynamic_process(data, fluids, fractions) 
    
    # Return a clean string to the LLM agent to prevent prompt bleeding
    return (
        "System Observation: The isochoric simulation was executed and the HTML graph was rendered successfully. "
        "Please present the following key results to the user in a natural, professional manner:\n\n"
        f"• Constant Molar Volume: {data['volumes_m3_mol'][0]:.6e} m³/mol\n"
        f"• Initial Pressure: {data['pressures_pa'][0]:.2e} Pa | Final Pressure: {data['pressures_pa'][-1]:.2e} Pa\n"
        f"• Initial Z-factor: {data['z_factors'][0]:.4f} | Final Z-factor: {data['z_factors'][-1]:.4f}\n"
        f"• Change in Residual Internal Energy (ΔU): {data['delta_U_J_mol'][-1]:.2f} J/mol\n"
        f"• Change in Residual Helmholtz Free Energy (ΔA): {data['delta_A_J_mol'][-1]:.2f} J/mol\n"
        f"• Change in Residual Entropy (ΔS): {data['delta_S_J_mol_K'][-1]:.4f} J/(mol·K)"
    )

# Combinamos las utilidades externas con las herramientas expertas locales
thermo_tools = conversion_tools_list + [
    tool_isothermal_process, 
    tool_isobaric_process, 
    tool_adiabatic_process,
    tool_isochoric_process
]

# =====================================================================
# AGENT SETUP
# =====================================================================
thermo_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Thermodynamic Specialist Agent. 
    You execute high-fidelity calculations using the Peng-Robinson Equation of State.
    
    Always execute steps in this STRICT order:
    1. If the user provides pressure, temperature, OR VOLUME in non-SI units, YOU MUST use the appropriate conversion tools to get exact values in Pascals (Pa), Kelvin (K), and cubic meters per mole (m^3/mol).
    2. Call the appropriate simulation tool (e.g., tool_isothermal_process, tool_isochoric_process) using ONLY the converted pure SI values.
    3. The simulation tool will automatically plot the data and return a text summary of the results to you.
    4. Present this summary to the user clearly and professionally using Markdown formatting (bullet points and bold text). 
    5. Briefly add a 1-2 sentence scientific interpretation of the key energy shifts (e.g., what the change in Gibbs, Helmholtz, or Entropy implies physically for the system). Do not make up numbers; use only what the tool returns."""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

thermo_agent = create_tool_calling_agent(llm, thermo_tools, thermo_prompt)
thermo_executor = AgentExecutor(agent=thermo_agent, tools=thermo_tools, verbose=True)

def route_to_thermodynamics(query: str) -> str:
    """AI TOOL: Routes extracted thermodynamic parameters to the Thermodynamic Specialist Agent."""
    return thermo_executor.invoke({"input": query})["output"]

thermo_specialist_tool = Tool(
    name="Thermodynamics_Specialist",
    func=route_to_thermodynamics,
    description=(
        "Use this tool ONLY when the user's problem involves real gases, "
        "the Peng-Robinson Equation of State (EOS), macroscopic thermodynamic "
        "trajectories (e.g., isobaric, isothermal, isochoric, adiabatic), or "
        "P-V-T state properties. You MUST pass the fully extracted problem details "
        "as the input, including fluid names in standard English, molar fractions, "
        "and exact state variables with their original units (including volume or molar volume if provided)."
    )
)