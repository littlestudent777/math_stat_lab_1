import os
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import norm, cauchy, uniform
from scipy.special import gamma

save_path = "distribution_plots"
os.makedirs(save_path, exist_ok=True)  # Create a directory for graphs, if it doesn't exist

# Distribution parameters and corresponding density functions as dictionary
distributions = {
    "Normal": {
        "params": {"mu": 0.0, "sigma": 1.0},
        "generator": lambda params, size: np.random.normal(params["mu"], params["sigma"], size),
        "density": lambda x, params: norm(params["mu"], params["sigma"]).pdf(x),
        # The number of histogram bins is approximately the square root of each sample size
        "bins": [3, 7, 10]
    },
    "Cauchy": {
        "params": {"x0": 0.0, "gamma": 1.0},
        "generator": lambda params, size: params["x0"] + params["gamma"] * np.random.standard_cauchy(size),
        "density": lambda x, params: cauchy(params["x0"], params["gamma"]).pdf(x),
        # For Cauchy distribution bins have to be bigger in order to avoid empty spaces in the histogram
        "bins": [2, 3, 4]
    },
    "Poisson": {
        "params": {"mu": 10.0},
        "generator": lambda params, size: np.random.poisson(params["mu"], size),
        "density": lambda x, params: (params["mu"] ** x) * np.exp(-params["mu"]) / gamma(x + 1),
        "bins": [3, 6, 9]
    },
    "Uniform": {
        "params": {"a": -sqrt(3), "b": sqrt(3)},
        "generator": lambda params, size: np.random.uniform(params["a"], params["b"], size),
        "density": lambda x, params: uniform(params["a"], params["b"] - params["a"]).pdf(x),
        "bins": [3, 7, 10]
    }
}


# Plots distribution graphs and histograms and saves them as png
def plot_distributions(sample_sizes: list[int]) -> None:

    # Iterate through all distributions
    for name, dist in distributions.items():
        # Extract distribution-specific parameters from the 'dist' dictionary
        params = dist["params"]

        # Create a new figure for the current distribution
        plt.figure(figsize=(15, 5))  # The overall size of the image is set

        plt.suptitle(f'{name} Distribution')

        # Iterate through the sample sizes
        for i, size in enumerate(sample_sizes):
            # Generate a random sample of needed size using the distribution's generator function
            sample = dist["generator"](params, size)

            # Make a grid
            x_min, x_max = min(sample), max(sample)
            x = np.linspace(x_min, x_max, 100)
            # Compute the values of the probability density function over x
            y = dist["density"](x, params)

            # Create a subplot for this sample size
            plt.subplot(1, len(sample_sizes), i + 1)

            # Plot the histogram of the sample
            plt.hist(sample, bins=dist["bins"][i], density=True,
                     color='white', edgecolor='black', alpha=0.7)

            # Plot the probability density function curve over the histogram
            plt.plot(x, y, color='black', linewidth=1)

            plt.title(f'n = {size}')
            plt.xlabel('Values')
            plt.ylabel('Density')

            # Special handling for the Cauchy distribution with large sample sizes
            # - The Cauchy distribution has a heavy tail, so the density range can vary significantly
            # - To better visualize the data, logarithmic y-axis is used for large samples
            if name == "Cauchy" and size >= 50:
                plt.yscale('log')
                plt.ylabel('log of density')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{name}.png"), dpi=300)

        plt.close()
