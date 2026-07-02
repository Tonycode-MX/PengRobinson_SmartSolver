import numpy as np
from typing import Dict, Any, List

class HydrogelAdvancedSorter:
    """
    Advanced dominance post-processor for sensitivity matrices.
    Applies a noise gate filter to eliminate insignificant parameters
    and evaluates the Pareto principle to isolate critical manufacturing factors.
    """
    def __init__(self, noise_threshold: float = 0.05):
        """
        Initializes the sorter with a baseline operational noise threshold.
        """
        self.noise_threshold = noise_threshold

    def calculate_dominance_matrix(self, raw_dependencies: Dict[str, float]) -> Dict[str, Any]:
        """
        Ingests a dictionary of Pearson coefficients, filters out noise,
        ranks elements by absolute magnitude, and flags the Pareto (80/20) driver set.
        
        Args:
            raw_dependencies (Dict[str, float]): Raw signed Pearson correlation r values.
            
        Returns:
            Dict[str, Any]: Ranked structure containing filtered items and Pareto metadata.
        """
        processed_elements = []
        total_absolute_impact = 0.0
        
        # Step 1: Filter through the noise threshold gate and compute absolute magnitudes
        for param_name, r_value in raw_dependencies.items():
            clean_name = param_name.replace("_influence", "")
            absolute_magnitude = abs(r_value)
            
            # Noise threshold gate condition check
            if absolute_magnitude < self.noise_threshold:
                is_significant = False
                status_label = "ignored_noise"
            else:
                is_significant = True
                status_label = "statistically_significant"
                
            # Determine the physical direction of the relationship based on the algebraic sign
            relationship = "direct" if r_value > 0 else "inverse"
            if r_value == 0:
                relationship = "neutral"
                
            element_data = {
                "parameter": clean_name,
                "raw_r_coefficient": round(r_value, 4),
                "absolute_impact": round(absolute_magnitude, 4),
                "relationship_type": relationship,
                "significance_status": status_label,
                "is_significant": is_significant
            }
            
            processed_elements.append(element_data)
            
            # Accumulate impact only for variables that clear the noise gate
            if is_significant:
                total_absolute_impact += absolute_magnitude

        # Step 2: Sort by absolute impact in descending order (Dominance Sorting)
        sorted_elements = sorted(
            processed_elements, 
            key=lambda item: item["absolute_impact"], 
            reverse=True
        )

        # Step 3: Apply the Pareto Principle (Cumulative 80% threshold calculation)
        cumulative_impact = 0.0
        pareto_drivers = []
        
        for element in sorted_elements:
            # If the variable was ignored due to noise, its Pareto contribution is zero
            if not element["is_significant"] or total_absolute_impact == 0:
                element["cumulative_percentage"] = 100.0
                element["belongs_to_pareto_80"] = False
                continue
                
            # Calculate individual and cumulative contribution percentages
            contribution_pct = (element["absolute_impact"] / total_absolute_impact) * 100
            cumulative_impact += contribution_pct
            
            element["cumulative_percentage"] = round(cumulative_impact, 2)
            
            # If within the 80% threshold (or if it is the first dominant element)
            if cumulative_impact <= 80.0 or len(pareto_drivers) == 0:
                element["belongs_to_pareto_80"] = True
                pareto_drivers.append(element["parameter"])
            else:
                element["belongs_to_pareto_80"] = False

        return {
            "total_active_weight": round(total_absolute_impact, 4),
            "primary_pareto_drivers": pareto_drivers if total_absolute_impact > 0 else [],
            "hierarchical_ranking": sorted_elements
        }

    def process_full_manifest(self, sensitivity_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates dominance rankings across separated structural physical outputs.
        """
        raw_indices = sensitivity_report["sensitivity_indices"]
        
        # Resolve dominance for Fiber Diameter (Level 4) and Swelling Ratio (Level 1)
        fiber_analysis = self.calculate_dominance_matrix(raw_indices["fiber_diameter_dependencies"])
        swelling_analysis = self.calculate_dominance_matrix(raw_indices["swelling_ratio_dependencies"])
        
        return {
            "status": "success",
            "samples_processed": sensitivity_report["samples_analyzed"],
            "analytics_manifest": {
                "fiber_diameter_control_path": fiber_analysis,
                "swelling_ratio_control_path": swelling_analysis
            }
        }


# =====================================================================
# STANDALONE PRODUCTION DOMINANCE TEST RUN (SANITY RUN)
# =====================================================================
if __name__ == "__main__":
    print("Launching integration test for the Advanced Dominance Sorter...")
    
    # Instantiate the classifier with a noise threshold of 0.05
    sorter = HydrogelAdvancedSorter(noise_threshold=0.05)
    
    # Simulate raw Pearson correlation report from stochastic computations
    mock_sensitivity_data = {
        "samples_analyzed": 10000,
        "sensitivity_indices": {
            "fiber_diameter_dependencies": {
                "voltage_kv_influence": -0.7845,       # Massive inverse impact
                "viscosity_pa_s_influence": 0.4512,     # Strong direct relationship
                "flow_rate_ml_h_influence": 0.1245,     # Mild direct relationship
                "temperature_celsius_influence": -0.0112 # Statistical noise (should be bypassed)
            },
            "swelling_ratio_dependencies": {
                "voltage_kv_influence": 0.0002,         # pure noise
                "viscosity_pa_s_influence": -0.0041,    # pure noise
                "flow_rate_ml_h_influence": 0.0011,     # pure noise
                "temperature_celsius_influence": -0.9912 # Absolute dominant thermodynamic shift
            }
        }
    }
    
    # Execute the sorting and Pareto analysis pipeline
    final_manifest = sorter.process_full_manifest(mock_sensitivity_data)
    
    print("\n🏆 [PARETO MANIFEST AND NOISE FILTER RESOLVED]:")
    
    # Validate results for Fiber Diameter 
    fiber_path = final_manifest["analytics_manifest"]["fiber_diameter_control_path"]
    print(f"   ➤ Primary Pareto Drivers for Fiber Matrix: {fiber_path['primary_pareto_drivers']}")
    for rank, item in enumerate(fiber_path["hierarchical_ranking"], start=1):
        print(f"      [{rank}] {item['parameter'].upper()} -> r: {item['raw_r_coefficient']} | Cumulative: {item['cumulative_percentage']}% | Status: {item['significance_status']} | Pareto: {item['belongs_to_pareto_80']}")
        
    # Validate results for Thermal Swelling 
    swelling_path = final_manifest["analytics_manifest"]["swelling_ratio_control_path"]
    print(f"\n   ➤ Primary Pareto Drivers for Swelling Performance: {swelling_path['primary_pareto_drivers']}")
    print(f"      Top Rank Item Specifications: {swelling_path['hierarchical_ranking'][0]}")