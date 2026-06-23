import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_thermodynamic_process(process_data: dict, fluids: list, fractions: list) -> None:
    """
    This function acts as a dynamic dispatcher, automatically detecting the type 
    of thermodynamic process (Isobaric, Isothermal, etc.) from the input dictionary 
    and overlaying the appropriate energetic properties (Enthalpy, Internal Energy, 
    Gibbs Free Energy, or Work) on top of the macroscopic molar volume profile.

    Args:
        process_data (dict): Standardized dictionary from the processes module 
                             containing trajectory vectors.
        fluids (list): List of fluid identifiers (e.g., ['Water', 'CarbonDioxide']).
        fractions (list): Molar fractions corresponding to each fluid component.

    Returns:
        None: This function does not return any values; it directly opens an 
              interactive Plotly dashboard in the default web browser.

    Raises:
        ValueError: If the 'fluids' and 'fractions' lists do not have the same length.
        KeyError: If required baseline keys are missing from 'process_data'.
    """
    # --- Input Validation ---
    if len(fluids) != len(fractions):
        raise ValueError("The number of fluids must match the number of molar fractions.")
        
    required_keys = ["process", "x_axis_name", "x_values", "z_factors", "volumes_m3_mol"]
    for key in required_keys:
        if key not in process_data:
            raise KeyError(f"Missing required key '{key}' in the process_data dictionary.")

    # 1. Extract core trajectory vectors from the backend dictionary
    process_name = process_data["process"]
    x_axis_backend = process_data["x_axis_name"]
    x_values = process_data["x_values"]
    z_factors = process_data["z_factors"]
    volumes = process_data["volumes_m3_mol"]
    
    # Localize the X-axis name for the user interface
    x_label_display = "Temperature (K)" if "Temperature" in x_axis_backend else "Pressure (Pa)"
    
    # 2. Dynamic secondary Y-axis variable mapping (New energetic properties)
    y2_values = None
    y2_label = ""
    y2_color = "forestgreen"
    y2_dash = "dash"
    
    if "delta_H_J_mol" in process_data:
        y2_values = process_data["delta_H_J_mol"]
        y2_label = "ΔH Residual Enthalpy (J/mol)"
        y2_color = "darkorange"
    elif "delta_U_J_mol" in process_data:
        y2_values = process_data["delta_U_J_mol"]
        y2_label = "ΔU Residual Internal Energy (J/mol)"
        y2_color = "darkorchid"
    elif "delta_G_J_mol" in process_data:
        y2_values = process_data["delta_G_J_mol"]
        y2_label = "ΔG Residual Gibbs Free Energy (J/mol)"
        y2_color = "mediumseagreen"
    elif "work_produced_J_mol" in process_data:
        y2_values = process_data["work_produced_J_mol"]
        y2_label = "Net Work Produced (J/mol)"
        y2_color = "crimson"

    # 3. Handle specific extra trajectories (e.g., Temperature drop in Adiabatic)
    extra_y2_values = None
    extra_y2_label = ""
    if "temperatures_K" in process_data and x_axis_backend == "Pressure (Pa)":
        extra_y2_values = process_data["temperatures_K"]
        extra_y2_label = "Actual Temperature (K)"

    # 4. Configure Plotly Figure Layout and Subplots
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=(
            f"Real Gas Behavior: Z Factor vs {x_label_display}", 
            f"Energetic and State Profiles vs {x_label_display}"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    # --- SUBPLOT 1: Microscopic Deviations (Compressibility Factor Z) ---
    fig.add_trace(go.Scatter(
        x=x_values, y=z_factors,
        mode='lines+markers', name='Actual Z Factor',
        line=dict(color='firebrick', width=3),
        hovertemplate=f'<b>{x_label_display}</b>: %{{x}}<br><b>Z Factor</b>: %{{y:.4f}}<extra></extra>'
    ), row=1, col=1)
    
    # --- SUBPLOT 2: Macroscopic State (Molar Volume) ---
    fig.add_trace(go.Scatter(
        x=x_values, y=volumes,
        mode='lines+markers', name='Molar Volume (V)',
        line=dict(color='royalblue', width=2),
        hovertemplate=f'<b>{x_label_display}</b>: %{{x}}<br><b>Volume</b>: %{{y:.6e}} m³/mol<extra></extra>'
    ), row=1, col=2, secondary_y=False)
    
    # Overlay the primary thermodynamic property line if present
    if y2_values is not None:
        fig.add_trace(go.Scatter(
            x=x_values, y=y2_values,
            mode='lines', name=y2_label,
            line=dict(color=y2_color, width=2, dash=y2_dash),
            hovertemplate=f'<b>{x_label_display}</b>: %{{x}}<br><b>{y2_label}</b>: %{{y:.2f}}<extra></extra>'
        ), row=1, col=2, secondary_y=True)

    # Overlay extra trajectories (e.g., Thermal drops in expansions)
    if extra_y2_values is not None:
        fig.add_trace(go.Scatter(
            x=x_values, y=extra_y2_values,
            mode='lines', name=extra_y2_label,
            line=dict(color='darkcyan', width=2, dash='dot'),
            hovertemplate=f'<b>{x_label_display}</b>: %{{x}}<br><b>{extra_y2_label}</b>: %{{y:.2f}} K<extra></extra>'
        ), row=1, col=2)

    # 5. Global Aesthetic Formatting (Plotly White Standard)
    mixture_format = ", ".join([f"{f} ({x*100:.1f}%)" for f, x in zip(fluids, fractions)])
    
    # Translate process names for user interface titles
    process_translation = {
        "Isobaric": "Isobaric",
        "Isothermal": "Isothermal",
        "Isochoric": "Isochoric",
        "Adiabatic": "Isentropic Adiabatic"
    }
    translated_name = process_translation.get(process_name, process_name)

    fig.update_layout(
        title=dict(
            text=f"<b>Advanced Peng-Robinson EOS Simulator — {translated_name} Process</b><br><sub>Multicomponent Mixture: {mixture_format}</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        template="plotly_white",
        legend=dict(
            orientation="h", 
            yanchor="bottom", y=1.05, 
            xanchor="center", x=0.5
        ),
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    # Axis configuration
    fig.update_xaxes(title_text=x_label_display, row=1, col=1)
    fig.update_xaxes(title_text=x_label_display, row=1, col=2)
    fig.update_yaxes(title_text="Z Factor (Dimensionless)", row=1, col=1)
    fig.update_yaxes(title_text="Molar Volume (m³/mol)", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Energetic Properties (J/mol)", row=1, col=2, secondary_y=True)
    #fig.update_yaxes(title_text="Actual Thermodynamic Properties", row=1, col=2)
    
    # Launch interactive dashboard in default web browser
    #fig.show()
    #fig.write_html("dashboard_termodinamico.html", auto_open=True, include_plotlyjs="cdn")
    fig.write_json("latest_plot.json")

# --- Standalone Testing Block ---
if __name__ == "__main__":
    print("📊 Executing standalone Plotter validation...")
    
    # Ensure this script is executed from the project root for the import to resolve
    from thermo_core.processes import simulate_adiabatic_process
    import numpy as np
    
    test_fluids = ["Water", "CarbonDioxide"]
    test_fractions = [0.75, 0.25]
    test_kij = np.zeros((2, 2))
    
    # Generate path data using pure SI units (Pascals) as enforced by the backend
    path_data = simulate_adiabatic_process(
        p_start=5000000.0,  
        p_end=1500000.0,    
        t_start=600.0,      
        fluids=test_fluids, 
        fractions=test_fractions, 
        kij_matrix=test_kij, 
        points=30
    )
    
    # Render interactive visualization
    plot_thermodynamic_process(path_data, test_fluids, test_fractions)