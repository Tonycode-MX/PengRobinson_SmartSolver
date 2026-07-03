import json
import numpy as np
from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.utils as utils

class HydrogelGrapher3D:
    """
    Advanced 3D analytical visualization generator for macromolecular hydrogels.
    Computes joint probability density distributions across process parameter gradients
    and exports high-performance interactive 3D mesh graphs in JSON format.
    """
    def __init__(self):
        """
        Initializes the 3D engine with high-contrast laboratory color spectrums.
        """
        self.colorscale_simulated = "Viridis"       # Professional gradient for simulation clouds
        self.color_experimental_mesh = "#ff7f0e"    # Safety orange for physical baseline anchors

    def generate_surface_distribution_3d_json(
        self, 
        simulated_diameters: List[float], 
        voltages_kv: List[float],
        experimental_diameters: List[float]
    ) -> str:
        """
        Processes stochastic multi-variable tracking arrays and superimposes a 3D surface
        mapping how fiber diameter distributions respond to custom physical voltage gradients.
        """
        # Convert raw stream lists safely into rigid multi-dimensional NumPy arrays
        sim_d = np.array(simulated_diameters, dtype=float)
        v_kv = np.array(voltages_kv, dtype=float)
        exp_d = np.array(experimental_diameters, dtype=float)

        # Compute a 2D histogram to derive a joint probability density grid matrix
        # This bins the interaction data into an explicit X-Y physical coordinate plane
        density_matrix, x_edges, y_edges = np.histogram2d(
            v_kv, sim_d, bins=[25, 25], density=True
        )

        # Calculate coordinate midpoints from edge arrays to construct structural meshgrids
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
        x_mesh, y_mesh = np.meshgrid(x_centers, y_centers)

        # Instantiate the main Plotly 3D Figure container
        fig = go.Figure()

        # Trace A: Render the Monte Carlo stochastic response cloud as a continuous 3D Surface
        fig.add_trace(go.Surface(
            x=x_mesh,
            y=y_mesh,
            z=density_matrix.T,  # Transpose matrix to properly align with row-major mesh axes
            name="Stochastic Simulation Mesh",
            colorscale=self.colorscale_simulated,
            colorbar=dict(
                title=dict(
                    text="Probability Density",
                    side="top"  # Correct modern property configuration mapping mapping
                ),
                thickness=15,
                len=0.6
            ),
            opacity=0.85
        ))

        # Trace B: Overlay discrete experimental control samples as a distinct 3D Scatter layer
        # Maps the actual microscope data at the current active baseline point for calibration
        fixed_baseline_voltage = np.mean(v_kv)
        fig.add_trace(go.Scatter3d(
            x=[fixed_baseline_voltage] * len(exp_d),
            y=exp_d,
            z=np.ones(len(exp_d)) * (np.max(density_matrix) * 0.1), # Offset points to hover visually
            mode='markers',
            name="Microscope Controls",
            marker=dict(
                size=5,
                color=self.color_experimental_mesh,
                opacity=0.9,
                symbol='diamond'
            )
        ))

        # Enforce strict spatial documentation axis formatting and layout aesthetics
        fig.update_layout(
            title={
                'text': "Level 5 Validation Matrix - 3D Surface Response Mapping",
                'y': 0.93,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            scene=dict(
                xaxis_title="Electrospinning Voltage (kV)",
                yaxis_title="Fiber Diameter (nm)",
                zaxis_title="Probability Density",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)  # Sets an optimal standard isometric viewing angle
                ),
                aspectmode='manual',
                aspectratio=dict(x=1.0, y=1.0, z=0.7) # Mutes Z axis compression for clarity
            ),
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.02,
                xanchor="center",
                x=0.5
            ),
            font=dict(
                family="Helvetica, Arial, sans-serif",
                size=11
            ),
            margin=dict(l=0, r=0, b=0, t=40) # Eliminates padding for dense embedded frontend layout blocks
        )

        # Programmatically dump internal states into a secure JSON packet for network routing
        serialized_json_graph_3d = json.dumps(fig, cls=utils.PlotlyJSONEncoder)
        return serialized_json_graph_3d


# =====================================================================
# MODULE STANDALONE INTEGRATION TEST MATRIX (3D ENGINE BENCHMARK)
# =====================================================================
if __name__ == "__main__":
    print("==============================================================")
    print("Initializing Hydrogel 3D Engine Standalone Environment Run...")
    print("==============================================================\n")
    
    # Initialize isolated dynamic sample streams mapping co-dependent variables
    rng = np.random.default_rng()
    samples_count = 8000
    
    # Generate mock process vectors simulating normal voltage noise and diameter response curves
    mock_voltages = rng.normal(loc=20.0, scale=1.5, size=samples_count).tolist()
    # Apply inverse relationship logic (higher voltage reduces fiber size)
    mock_diameters = (150.0 + (500.0 / np.array(mock_voltages)) + rng.normal(0, 8, samples_count)).tolist()
    mock_lab_microscope = rng.normal(loc=175.2, scale=9.1, size=40).tolist()
    
    # Instantiate the visualization engine
    graph_engine_3d = HydrogelGrapher3D()
    
    # Execute serialization to check schema state stability
    output_json_3d = graph_engine_3d.generate_surface_distribution_3d_json(
        mock_diameters,
        mock_voltages,
        mock_lab_microscope
    )
    
    print("[SUCCESS] 3D Surface Matrix visualization finalized successfully.")
    print(f"[INFO] Serialized Plotly 3D JSON payload length: {len(output_json_3d)} characters.")