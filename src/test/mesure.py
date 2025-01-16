import subprocess
import json

problem_sizes = [10, 50, 100, 200, 500, 1000, 2000]
implementations = ["naive","optim", "optim_v2", "simd", "simd_optim", "gpu", "mipp", "mipp_v2", "mixed", "barnes_hut"]
runs_per_implementation = 5  # Number of runs to minimize noise
iterations = 100  # Fixed number of iterations per run

# Results storage
results = {}


for size in problem_sizes:
    results[size] = {}
    for impl in implementations:
        times = []
        fps_values = []

        for run in range(runs_per_implementation):
            try:
                # Command to execute the binary with the given parameters
                command = [
                    "./bin/murb",
                    "-n", str(size),
                    "-i", str(iterations),
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
            # Record the minimum time and corresponding maximum FPS
            results[size][impl] = {
                "time": min(times),
                "fps": max(fps_values),
            }

# Determine the best implementation for each problem size
best_implementations = {}
for size, data in results.items():
    best_impl = min(data.items(), key=lambda x: x[1]["time"])
    best_implementations[size] = {
        "implementation": best_impl[0],
        "time": best_impl[1]["time"],
        "fps": best_impl[1]["fps"],
    }

# Output results
print("Best Implementations by Problem Size:")
for size, info in best_implementations.items():
    print(f"Size {size}: Best Implementation = {info['implementation']}, Time = {info['time']} ms, FPS = {info['fps']}")

# Save results to a JSON file
with open("benchmark_results.json", "w") as f:
    json.dump({"results": results, "best": best_implementations}, f, indent=4)
