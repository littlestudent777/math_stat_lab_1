import numpy as np
import pandas as pd
# from IPython.display import display


# Calculates statistical characteristics for a sample
def calculate_statistics(sample: np.ndarray) -> np.ndarray:
    size = len(sample)
    # Sort the sample
    sample_sorted = np.sort(sample)

    # Mean
    x = np.mean(sample_sorted)

    # Median
    med_x = np.median(sample_sorted)

    z_q = (sample_sorted[int(np.ceil(size / 4.0) - 1)] +
           sample_sorted[int(np.ceil(3.0 * size / 4.0) - 1)]) / 2.0

    return np.array([x, med_x, z_q])


# Function for formatting numbers depending on the index of the string
def format_rows(val: float, index: str) -> str:
    if index == "E(z)^{(4)}":
        return f"{val:.2g}"  # Two significant figures for E(z)
    elif index == "D(z)^{(5)}":
        return f"{val:.4g}"  # Four significant figures for D(z)


# Calculates and outputs statistical characteristics for different sample types
def print_characteristics(sample_size: list[int]) -> None:
    # A dictionary of sample generators and their names
    distributions = {
        "Normal": lambda n: np.random.normal(loc=0.0, scale=1.0, size=n),
        "Cauchy": lambda n: np.random.standard_cauchy(size=n),
        "Poisson": lambda n: np.random.poisson(lam=10.0, size=n),
        "Uniform": lambda n: np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=n)
    }

    repeats = 1000

    for name, method in distributions.items():
        print("\\begin{center}\n" + f"\\textbf{{{name + ' distribution'}}}")

        for size in sample_size:
            data = np.zeros([2, 3])  # To store mean and variance values

            for i in range(repeats):
                sample = method(size)  # Generate a sample
                stats = calculate_statistics(sample)  # Calculate statistics for the sample

                data[0] += stats  # Add to the sum for means
                data[1] += stats ** 2  # Add square for variance calculation

            means = data[0] / repeats  # Calculate means
            variances = (data[1] / repeats) - (means ** 2)  # Calculate variances

            # Create and display a DataFrame containing statistics
            df = pd.DataFrame(
                [means, variances],
                columns=["x^{(1)}", "med x^{(2)}", "z_Q^{(3)}"],
                index=["E(z)^{(4)}", "D(z)^{(5)}"]
            )

            # Apply formatting to each row of the DataFrame
            formatted_df = df.apply(lambda row: row.apply(format_rows, index=row.name), axis=1)

            latex_str = formatted_df.to_latex(
                index=True,
                caption=f'Sample size {size}',
                escape=False
            )

            latex_str = latex_str.replace('\\begin{table}', '\\begin{table}[H] \\centering')
            latex_str = latex_str.replace('\\caption{', '\\caption*{')
            print(latex_str)

        print('\\end{center}\n')
