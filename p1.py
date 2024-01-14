import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.optimize import curve_fit

def fit_weibull_to_data(data):
    """Fit Weibull distribution parameters to data."""
    shape, loc, scale = weibull_min.fit(data)
    return shape, scale

def project_waste_with_supporting_points(initial_capacity, degradation_rate, supporting_points, years, weight_per_capacity):
    """Project cumulative waste in tonnes with supporting points for early-loss scenario."""
    waste = np.zeros_like(years, dtype=float)
    cumulative_capacity = initial_capacity

    for i, year in enumerate(years):
        if year in supporting_points:
            # If the current year has a supporting point, reduce the capacity based on the specified percentage loss
            cumulative_capacity *= (1 - supporting_points[year])
        else:
            # If no supporting point for the current year, apply the regular degradation rate
            cumulative_capacity *= np.exp(-degradation_rate * (year - years[0]))

        # Calculate waste in tonnes based on the reduction in capacity and weight per capacity
        waste[i] = (initial_capacity - cumulative_capacity) * weight_per_capacity

    return np.cumsum(waste)

def main():
    # Provided data
    years_data = np.array([2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050])
    installed_capacity_data = np.array([222, 511, 954, 1632, 2225, 2895, 3654, 4512])
    weight_per_capacity = 0.02  # Example: 0.02 tonnes per installed capacity megawatt

    # Regular-loss scenario
    shape_regular, scale_regular = fit_weibull_to_data(installed_capacity_data)
    degradation_rate_regular_loss = 1 / scale_regular  # Inverse of scale is the degradation rate

    # Early-loss scenario parameters (using provided data for simplicity)
    supporting_points = {
        2020: 0.005,  # Example: Installation/transport damages within the first 2 years
        2030: 0.02,   # Example: After 10 years
        2035: 0.04    # Example: After 15 years
    }
    shape_early, scale_early = fit_weibull_to_data(installed_capacity_data)
    degradation_rate_early_loss = 1 / scale_early  # Inverse of scale is the degradation rate

    # Project cumulative waste in tonnes for each scenario
    years_projection = np.arange(2015, 2051)  # Extend the projection to 2050
    waste_early_loss = project_waste_with_supporting_points(installed_capacity_data[0], degradation_rate_early_loss, supporting_points, years_projection, weight_per_capacity)
    waste_regular_loss = project_waste_with_supporting_points(installed_capacity_data[0], degradation_rate_regular_loss, {}, years_projection, weight_per_capacity)

    # Plot results
    plt.plot(years_projection, waste_early_loss, label='Early Loss Scenario', color='blue')
    plt.plot(years_projection, waste_regular_loss, label='Regular Loss Scenario', color='orange')
    plt.xlabel('Years')
    plt.ylabel('Cumulative Waste Projection (Tonnes)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
