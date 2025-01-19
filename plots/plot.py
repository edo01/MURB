import json
import matplotlib.pyplot as plt

# Load the data from a JSON file
with open("results_30000.json", "r") as file:  # Replace 'data.json' with your file path
    data = json.load(file)

def plot_metric(data, metric="fps"):
    """
    Plots the specified metric ('time' or 'fps') for each implementation.
    
    :param data: Dictionary containing the data.
    :param metric: Metric to plot on the y-axis ('time' or 'fps').
    """
    problem_sizes = sorted(map(int, data.keys()))  # Get problem sizes as sorted integers
    implementations = list(next(iter(data.values())).keys())  # Get implementation names

    # Prepare the plot
    plt.figure(figsize=(10, 6))
    
    # Plot data for each implementation
    for impl in implementations:
        y_values = [data[str(size)][impl][metric] for size in problem_sizes if impl in data[str(size)]]
        # Reduce problem sizes to the ones that have the implementation
        plot_problem_sizes = [size for size in problem_sizes if impl in data[str(size)]]
        plt.plot(plot_problem_sizes, y_values, label=impl, marker="o")

    # Customize plot
    plt.title(f"{metric.capitalize()} vs Problem Size")
    plt.xlabel("Problem Size")
    plt.ylabel(metric.capitalize())
    plt.xscale("log")  # Set x-axis to log scale
    plt.yscale("log")  # Set y-axis to log scale
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend(title="Implementation", loc="best")
    plt.tight_layout()

    # Show plot
    plt.show()

# Example usage: Plot 'time' or 'fps'
plot_metric(data, metric="fps")  # Change to 'fps' for FPS plot