from typing import Any, Dict, List, Union
import numpy as np


class HydrogelDataCompressor:
    """Optimizes API payload structures by compressing high-density Monte Carlo arrays.

    Transforms raw, heavy simulation distributions into lightweight histogram
    coordinate maps representing Probability Density Functions (PDF).
    """

    def __init__(self, default_bins: int = 30):
        """Initializes the compression engine with a target bin resolution.

        Args:
            default_bins (int, optional): Quantization interval count used to
                cluster continuous raw numeric values. Defaults to 30.
        """
        self.default_bins = default_bins

    def compress_distribution(
        self, raw_data: Union[List[float], np.ndarray], bins: int = None
    ) -> Dict[str, List[float]]:
        """Converts a high-density raw array of floats into compressed histogram points.

        Calculates localized bin centers and normalized probability density values:
        $$y_{\\text{density}} = \\frac{\\text{counts}}{N \\cdot \\Delta w}$$

        Where:
        - $N$ is the total sample count.
        - $\\Delta w$ is the width of the interval.
        - The integral over the entire compressed surface area equals 1.0.

        The structural coordinate coordinates for rendering maps follow:
        $$x_{\\text{center}} = \\frac{e_i + e_{i+1}}{2.0}$$

        Where $e_i$ and $e_{i+1}$ define the geometric edges of bin $i$.

        Args:
            raw_data (Union[List[float], np.ndarray]): Raw data grid containing
                thousands of stochastic simulation coordinates.
            bins (int, optional): Discrete number of intervals to construct.
                Defaults to instance baseline.

        Returns:
            Dict[str, List[float]]: Compressed visualization layout metrics:
                - "x_centers": Truncated floating-point values of bin centers.
                - "y_density": Density values representing the likelihood area.
        """
        target_bins = bins if bins is not None else self.default_bins
        data_array = np.array(raw_data, dtype=float)

        # Guard against empty arrays or invariant datasets (variance = 0)
        # np.ptp computes maximum - minimum value across the array slice
        if data_array.size == 0 or np.ptp(data_array) == 0.0:
            return {"x_centers": [], "y_density": []}

        # Compute the histogram using continuous probability density normalization
        counts, bin_edges = np.histogram(
            data_array, bins=target_bins, density=True
        )

        # Derive bin centers from edges to provide direct coordinate mapping
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        return {
            "x_centers": np.round(bin_centers, 3).tolist(),
            "y_density": np.round(counts, 6).tolist(),
        }

    def optimize_stochastic_payload(
        self, raw_monte_carlo_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Intercepts massive simulation payloads and strips down heavy array distributions.

        Replaces extensive raw item logs with compressed coordinate structures
        to minimize bandwidth consumption during API exchanges.

        Args:
            raw_monte_carlo_output (Dict[str, Any]): Uncompressed payload manifest
                containing raw simulation arrays under "raw_distribution_data_pures".

        Returns:
            Dict[str, Any]: Lightweight payload conforming to web transmission standards.

        Raises:
            KeyError: If mandatory source dictionary tracking keys are missing.
        """
        if "raw_distribution_data_pures" not in raw_monte_carlo_output:
            raise KeyError(
                "Payload missing mandatory 'raw_distribution_data_pures' key structure."
            )

        raw_distributions = raw_monte_carlo_output[
            "raw_distribution_data_pures"
        ]

        # Compress hydrogel output data vectors independently
        compressed_fiber = self.compress_distribution(
            raw_distributions["fiber_diameter_nm"]
        )
        compressed_swelling = self.compress_distribution(
            raw_distributions["swelling_ratio"]
        )

        # Reconstruct the payload contract preserving metadata while protecting network bandwidth
        optimized_payload = {
            "status": raw_monte_carlo_output.get("status", "success"),
            "samples_computed": raw_monte_carlo_output.get(
                "samples_analyzed", 0
            ),
            "sensitivity_indices": raw_monte_carlo_output.get(
                "sensitivity_indices", {}
            ),
            "compressed_visualizations": {
                "fiber_diameter_pdf": compressed_fiber,
                "swelling_ratio_pdf": compressed_swelling,
            },
        }

        return optimized_payload


# =====================================================================
# STANDALONE PERFORMANCE COMPRESSION BENCHMARK (SANITY CHECK)
# =====================================================================
if __name__ == "__main__":
    print("Launching API network bandwidth compression simulation test...")
    compressor = HydrogelDataCompressor(default_bins=30)

    # Simulating a massive 10,000 sample output dataset coming from level 3 scripts
    np.random.seed(42)
    mock_large_fiber_array = np.random.normal(220.0, 15.5, 10000).tolist()
    mock_large_swelling_array = np.random.normal(2.1, 0.3, 10000).tolist()

    mock_input_payload = {
        "status": "success",
        "samples_analyzed": 10000,
        "sensitivity_indices": {
            "fiber_diameter_dependencies": {"voltage_kv_influence": -0.84},
            "swelling_ratio_dependencies": {
                "temperature_celsius_influence": -0.98
            },
        },
        "raw_distribution_data_pures": {
            "fiber_diameter_nm": mock_large_fiber_array,
            "swelling_ratio": mock_large_swelling_array,
        },
    }

    try:
        # Run the structural optimization pipeline
        optimized_output = compressor.optimize_stochastic_payload(
            mock_input_payload
        )

        print("\n📦 [NETWORK OPTIMIZATION BENCHMARK REPORT]:")
        print(
            f"   ➤ Original Array Size: {len(mock_large_fiber_array)} floats per variable."
        )

        fiber_coords = optimized_output["compressed_visualizations"][
            "fiber_diameter_pdf"
        ]
        print(
            f"   ➤ Compressed X-Axis Points Sent: {len(fiber_coords['x_centers'])}"
        )
        print(
            f"   ➤ Compressed Y-Axis Points Sent: {len(fiber_coords['y_density'])}"
        )

        # Calculate byte-size proxy reduction
        original_element_count = 10000 * 2
        compressed_element_count = len(fiber_coords["x_centers"]) * 4
        reduction_percentage = (
            1.0 - (compressed_element_count / original_element_count)
        ) * 100
        print(
            f"   ➤ Net Data Payload Reduction Rate: {reduction_percentage:.2f}%"
        )

    except KeyError as parse_error:
        print(f"❌ Optimization runtime failed: {parse_error}")