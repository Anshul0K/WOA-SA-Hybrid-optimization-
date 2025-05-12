import numpy as np
import os
import pandas as pd
import opfunu

# === Dragonfly Algorithm (DA) ===
def dragonfly_algorithm(objective_function, lower_bound, upper_bound, max_evals, pop_size, dim):
    eval_counter = [0]  # Use mutable list for evaluation count

    def wrapped_func(x):
        if eval_counter[0] >= max_evals:
            raise StopIteration
        eval_counter[0] += 1
        return objective_function(x)

    # Initialize dragonflies
    dragonflies = np.random.uniform(lower_bound, upper_bound, (pop_size, dim))
    velocities = np.random.uniform(-0.1 * (upper_bound - lower_bound), 0.1 * (upper_bound - lower_bound), (pop_size, dim))  # Initialize velocities
    fitness = np.array([wrapped_func(dragonfly) for dragonfly in dragonflies])

    # Find initial best dragonfly
    best_index = np.argmin(fitness)
    best_position = dragonflies[best_index].copy()
    best_fitness = fitness[best_index]

    # DA parameters
    s = 0.1  # Separation weight
    a = 0.1  # Alignment weight
    c = 0.1  # Cohesion weight
    f = 1.0  # Food attraction weight
    e = 1.0  # Enemy avoidance weight
    w = 0.9  # Inertia weight
    t = 0 # Iteration counter

    try:
        while eval_counter[0] < max_evals:
            t += 1
            for i in range(pop_size):
                S = np.zeros(dim)
                A = np.zeros(dim)
                C = np.zeros(dim)
                F = np.zeros(dim)
                E = np.zeros(dim)

                # Calculate separation
                for j in range(pop_size):
                    if i != j:
                        dist = np.linalg.norm(dragonflies[i] - dragonflies[j])
                        S += (dragonflies[i] - dragonflies[j]) / dist if dist > 0 else 0
                S = -S

                # Calculate alignment
                for j in range(pop_size):
                    if i != j:
                        A += velocities[j]
                A = A / (pop_size - 1) if pop_size > 1 else np.zeros(dim)

                # Calculate cohesion
                for j in range(pop_size):
                    if i != j:
                        C += dragonflies[j]
                C = (C / (pop_size - 1) if pop_size > 1 else np.zeros(dim)) - dragonflies[i]

                # Calculate food attraction
                F = best_position - dragonflies[i]

                # Calculate enemy avoidance (simplified - assumes one enemy at the edge of search space)
                E = np.random.uniform(lower_bound, upper_bound, dim) - dragonflies[i]

                # Update weights
                w = 0.9 - t * ((0.9 - 0.4) / max_evals) # Linearly decreasing inertia

                if np.linalg.norm(dragonflies[i] - best_position) < 0.1: # Threshold for food attraction.
                    velocities[i] = (A * a + C * c + F * f) + w * velocities[i]
                    dragonflies[i] += velocities[i]
                else:
                    velocities[i] = (S * s + A * a + C * c + E * e + w * velocities[i])
                    dragonflies[i] += velocities[i]

                # Clip dragonflies to the search space boundaries
                dragonflies[i] = np.clip(dragonflies[i], lower_bound, upper_bound)

                # Evaluate fitness
                fitness[i] = wrapped_func(dragonflies[i])

            # Update the best dragonfly
            current_best_index = np.argmin(fitness)
            if fitness[current_best_index] < best_fitness:
                best_fitness = fitness[current_best_index]
                best_position = dragonflies[current_best_index].copy()

    except StopIteration:
        pass

    return best_fitness



# === Benchmark Runner for CEC Benchmarks (2014, 2017, 2020, and 2022) ===
def run_cec_benchmark_da(cec_funcs, year="CEC-2020", dim=30, runs=50, evals=1200, pop_size=100, output_dir="cec_da_results"):
    os.makedirs(output_dir, exist_ok=True)
    results_dict = {}

    for func_class in cec_funcs:
        func_name = func_class.__name__
        print(f"Running DA on {func_name} from {year} with dimension {dim}...")

        try:
            func_obj = func_class(ndim=dim)
            if hasattr(func_obj, 'dim_supported'):
                if dim not in func_obj.dim_supported:
                    print(f"Skipping {func_name} from {year}: Dimension {dim} is not supported (supported: {func_obj.dim_supported}).")
                    continue
            else:
                print(f"Warning: {func_name} from {year} does not have 'dim_supported' attribute. Assuming dimension {dim} is acceptable.")
        except FileNotFoundError as e:
            print(f"Skipping {func_name} from {year}: File not found - {e}")
            continue
        except ValueError as e:
            print(f"Skipping {func_name} from {year}: Initialization error - {e}")
            continue
        except Exception as e:
            print(f"Skipping {func_name} from {year}: An unexpected error occurred during initialization - {e}")
            continue

        lower = func_obj.bounds[0][0]
        upper = func_obj.bounds[0][1]

        run_results = []
        for i in range(runs):
            try:
                result = dragonfly_algorithm(func_obj.evaluate, lower, upper, evals, pop_size, dim)
                run_results.append(result)
            except Exception as e:
                print(f"Error during optimization of {func_name} (Run {i+1}): {e}")
                run_results.append(np.nan)

        results_dict[func_name] = run_results

    # Create DataFrame and save as CSV
    df = pd.DataFrame.from_dict(results_dict, orient='index',
                                 columns=[f'Run {i + 1}' for i in range(runs)])
    csv_path = os.path.join(output_dir, f'{year}_dim_{dim}_da_results.csv')
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")



# === Run CEC Benchmarks with DA ===
if __name__ == "__main__":
    common_dim = 30
    cec2022_dim = 20
    pop_size = 100
    max_evaluations = 1200 * 5

    run_cec_benchmark_da(opfunu.get_functions_based_classname("2014"), year="CEC-2014", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_da_results")
    run_cec_benchmark_da(opfunu.get_functions_based_classname("2017"), year="CEC-2017", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_da_results")
    run_cec_benchmark_da(opfunu.get_functions_based_classname("2020"), year="CEC-2020", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_da_results")
    run_cec_benchmark_da(opfunu.get_functions_based_classname("2022"), year="CEC-2022", dim=cec2022_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_da_results")

    print("DA benchmark runs completed.")
