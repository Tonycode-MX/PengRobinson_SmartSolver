import numpy as np
from typing import List
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool, tool

from agents.llm_setup import llm

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
    temp_val: float,
    temp_unit: str,
    p_start_val: float,
    p_start_unit: str,
    p_end_val: float,
    p_end_unit: str,
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
        temp_val (float): The numerical value of the constant absolute temperature.
        temp_unit (str): The unit of the temperature (e.g., 'C', 'F', 'K', 'R').
        p_start_val (float): The numerical value of the initial absolute pressure.
        p_start_unit (str): The unit of the initial pressure (e.g., 'Pa', 'atm', 'psi', 'bar', 'MPa').
        p_end_val (float): The numerical value of the final absolute pressure.
        p_end_unit (str): The unit of the final pressure (e.g., 'Pa', 'atm', 'psi', 'bar', 'MPa').
        fluids (List[str]): List of valid fluid names (e.g., ['Methane', 'Ethane']).
        fractions (List[float]): List of molar fractions corresponding to the fluids (must sum to 1.0).
        
    Returns:
        str: A formatted, data-only summary of the initial and final thermodynamic states.
    """
    # Initialize a zero-matrix for binary interaction parameters (k_ij)
    kij = np.zeros((len(fluids), len(fluids)))
    
    # Execute the backend mathematical simulation with separated values and units
    data = simulate_isothermal_process(
        temp_val, temp_unit, 
        p_start_val, p_start_unit, 
        p_end_val, p_end_unit, 
        fluids, fractions, kij
    )
    
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
    press_val: float,
    press_unit: str,
    t_start_val: float,
    t_start_unit: str,
    t_end_val: float,
    t_end_unit: str,
    fluids: List[str], 
    fractions: List[float]
) -> str:
    """
    Simulates an isobaric (constant pressure) thermodynamic process and automatically generates an interactive plot.
    
    This tool computes the Peng-Robinson Equation of State (EOS) heating or cooling trajectory 
    for a real gas mixture at a constant pressure across a temperature gradient. It renders 
    an HTML dashboard and returns the key physical property shifts to the agent.
    
    Args:
        press_val (float): The numerical value of the constant absolute pressure.
        press_unit (str): The unit of the pressure (e.g., 'Pa', 'atm', 'psi', 'bar', 'MPa').
        t_start_val (float): The numerical value of the initial absolute temperature.
        t_start_unit (str): The unit of the initial temperature (e.g., 'C', 'F', 'K', 'R').
        t_end_val (float): The numerical value of the final absolute temperature.
        t_end_unit (str): The unit of the final temperature (e.g., 'C', 'F', 'K', 'R').
        fluids (List[str]): List of valid fluid names (e.g., ['Methane', 'Ethane']).
        fractions (List[float]): List of molar fractions corresponding to the fluids (must sum to 1.0).
        
    Returns:
        str: A formatted, data-only summary of the initial and final thermodynamic states.
    """
    # Initialize a zero-matrix for binary interaction parameters (k_ij)
    kij = np.zeros((len(fluids), len(fluids)))
    
    # Execute the backend mathematical simulation with the separated values and units
    data = simulate_isobaric_process(
        press_val, press_unit, 
        t_start_val, t_start_unit, 
        t_end_val, t_end_unit, 
        fluids, fractions, kij
    )
    
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
    p_start_val: float,
    p_start_unit: str,
    p_end_val: float,
    p_end_unit: str,
    t_start_val: float,
    t_start_unit: str,
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
        p_start_val (float): The numerical value of the initial absolute pressure.
        p_start_unit (str): The unit of the initial pressure (e.g., 'Pa', 'atm', 'psi', 'bar', 'MPa').
        p_end_val (float): The numerical value of the final absolute pressure.
        p_end_unit (str): The unit of the final pressure (e.g., 'Pa', 'atm', 'psi', 'bar', 'MPa').
        t_start_val (float): The numerical value of the initial absolute temperature.
        t_start_unit (str): The unit of the initial temperature (e.g., 'C', 'F', 'K', 'R').
        fluids (List[str]): List of valid fluid names (e.g., ['Methane', 'Ethane']).
        fractions (List[float]): List of molar fractions corresponding to the fluids (must sum to 1.0).
        
    Returns:
        str: A formatted, data-only summary of the initial and final thermodynamic states, including work and energy shifts.
    """
    # Initialize a zero-matrix for binary interaction parameters (k_ij)
    kij = np.zeros((len(fluids), len(fluids)))
    
    # Execute the backend mathematical simulation with separated values and units
    data = simulate_adiabatic_process(
        p_start_val, p_start_unit, 
        p_end_val, p_end_unit, 
        t_start_val, t_start_unit, 
        fluids, fractions, kij
    )
    
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
    vol_val: float,
    vol_unit: str,
    t_start_val: float,
    t_start_unit: str,
    t_end_val: float,
    t_end_unit: str,
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
        vol_val (float): The numerical value of the constant molar volume.
        vol_unit (str): The unit of the molar volume (e.g., 'm^3/mol', 'L/mol', 'cm^3/mol').
        t_start_val (float): The numerical value of the initial absolute temperature.
        t_start_unit (str): The unit of the initial temperature (e.g., 'C', 'F', 'K', 'R').
        t_end_val (float): The numerical value of the final absolute temperature.
        t_end_unit (str): The unit of the final temperature (e.g., 'C', 'F', 'K', 'R').
        fluids (List[str]): List of valid fluid names (e.g., ['Methane', 'Ethane']).
        fractions (List[float]): List of molar fractions corresponding to the fluids (must sum to 1.0).
        
    Returns:
        str: A formatted, data-only summary of the initial and final thermodynamic states.
    """
    # Initialize a zero-matrix for binary interaction parameters (k_ij)
    kij = np.zeros((len(fluids), len(fluids)))
    
    # Execute the backend mathematical simulation with separated values and units
    data = simulate_isochoric_process(
        vol_val, vol_unit, 
        t_start_val, t_start_unit, 
        t_end_val, t_end_unit, 
        fluids, fractions, kij
    )
    
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
thermo_tools = [
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
    1. Read the extracted problem details provided by the Orchestrator, which includes numerical values and their exact units.
    2. Call the appropriate simulation tool (e.g., tool_isothermal_process, tool_isochoric_process) passing the EXACT numerical values and unit strings provided. DO NOT ATTEMPT TO CONVERT UNITS.
    3. The backend simulation will automatically convert the units, plot the data, and return a text summary of the results to you.
    4. Present this summary to the user clearly and professionally using Markdown formatting (bullet points and bold text). 
    5. Briefly add a 1-2 sentence scientific interpretation of the key energy shifts (e.g., what the change in Gibbs, Helmholtz, or Entropy implies physically for the system). Do not make up numbers; use only what the tool returns."""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

thermo_agent = create_tool_calling_agent(llm, thermo_tools, thermo_prompt)
thermo_executor = AgentExecutor(agent=thermo_agent, tools=thermo_tools, verbose=False)

@tool(return_direct=True)
def thermo_specialist_tool(query: str) -> str:
    """
    Use this tool ONLY when the user's problem involves real gases, 
    the Peng-Robinson Equation of State (EOS), macroscopic thermodynamic 
    trajectories (e.g., isobaric, isothermal, isochoric, adiabatic), or 
    P-V-T state properties. You MUST pass the fully extracted problem details 
    as the input, including fluid names in standard English, molar fractions, 
    and exact state variables with their original units explicitly separated (e.g. Value: 450, Unit: 'F').
    """
    return thermo_executor.invoke({"input": query})["output"]