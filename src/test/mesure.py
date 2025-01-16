import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt


problem_sizes = [ 50, 100, 200, 500, 1000, 2000,3000]
implementations = ["cpu+naive", "optim", "simd", "gpu", "mipp", "omp"]
runs_per_implementation = 5  # Number of runs to minimize noise


results = {size: {} for size in problem_sizes}

for size in problem_sizes:
    for impl in implementations:
        times = []
        fps_values = []

        for run in range(runs_per_implementation):
            try:
                command = [
                    "./bin/murb", 
                    "-n", str(size),
                    "-i", "1000",  
                    "-v",
                    "-im", impl
                ]

                # Run the command and capture the output
                process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Parse the output to extract time and FPS
                output = process.stdout
                elapsed_time = None
                fps = None
                for line in output.splitlines():
                    if "Entire simulation took" in line:
                        parts = line.split()
                        elapsed_time = float(parts[3])  # Extract time in milliseconds
                        fps = float(parts[4][1:])  # Extract FPS (inside parentheses)
                        break

                if elapsed_time and fps:
                    times.append(elapsed_time)
                    fps_values.append(fps)

            except Exception as e:
                print(f"Error running implementation {impl} for size {size}: {e}")

        if times and fps_values:
            results[size][impl] = {
                "time": min(times),
                "fps": max(fps_values),
            }

# Save results to a JSON file
with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Find the best implementation for each problem size
best_implementations = {size: min(data.items(), key=lambda x: x[1]["time"])[0] for size, data in results.items()}

# Save best implementations to a JSON file
with open("best_implementations.json", "w") as f:
    json.dump(best_implementations, f, indent=4)


def plot_best_implementations():
    with open("best_implementations.json", "r") as f:
        best_impls = json.load(f)

    with open("benchmark_results.json", "r") as f:
        results = json.load(f)

    best_times = [results[size][best_impls[str(size)]]['time'] for size in problem_sizes]

    plt.figure(figsize=(12, 6))
    plt.bar(problem_sizes, best_times, tick_label=[best_impls[str(size)] for size in problem_sizes])
    plt.xlabel("Problem Size")
    plt.ylabel("Time (ms)")
    plt.title("Best Implementation for Each Problem Size")
    plt.grid(axis="y")
    plt.savefig("best_implementation_graph.png")
    plt.show()


plot_best_implementations()
