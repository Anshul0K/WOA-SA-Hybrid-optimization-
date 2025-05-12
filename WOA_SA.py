#WOA+SA
import numpy as np
import os
import pandas as pd
import opfunu

# === Modified Whale Optimization Algorithm with Simulated Annealing ===
def mod_woa_sa(objective_function, lower_bound, upper_bound, max_evals, num_whales, dim):
    eval_counter = [0]  # Use mutable list to allow updates inside nested scope

    def wrapped_func(x):
        if eval_counter[0] >= max_evals:
            raise StopIteration
        eval_counter[0] += 1
        return objective_function(x)

    whales = np.random.uniform(lower_bound, upper_bound, (num_whales, dim))
    fitness = np.array([wrapped_func(w) for w in whales])

    best_index = np.argmin(fitness)
    best_whale = whales[best_index].copy()
    best_fitness = fitness[best_index]

    try:
        while eval_counter[0] < max_evals:
            t = eval_counter[0] / max_evals
            r = 0.1 + 0.9 * t
            p = 0.9 - 0.8 * t
            pop_size = int(num_whales + 5 * np.sin(np.pi * t))
            pop_size = max(5, min(50, pop_size))

            new_whales = []
            for _ in range(pop_size):
                if np.random.rand() < p:
                    rand_pos = np.random.uniform(lower_bound, upper_bound, dim)
                    new_pos = rand_pos + r * np.random.randn(dim)
                else:
                    alpha = np.random.uniform(0.1, 0.5)
                    direction = best_whale - np.random.uniform(lower_bound, upper_bound, dim)
                    new_pos = best_whale + alpha * direction + r * np.random.randn(dim)

                new_pos = np.clip(new_pos, lower_bound, upper_bound)
                new_whales.append(new_pos)

            whales = np.array(new_whales)
            fitness = np.array([wrapped_func(w) for w in whales])
            current_best_index = np.argmin(fitness)

            if fitness[current_best_index] < best_fitness:
                best_fitness = fitness[current_best_index]
                best_whale = whales[current_best_index].copy()

    except StopIteration:
        pass

    return best_fitness


# === Benchmark Runner for CEC Benchmarks (2014, 2017, 2020, and 2022) ===
def run_cec_benchmark(cec_funcs, year="CEC-2020", dim=30, runs=50, evals=1200, num_whales=15, output_dir="cec_results"):
    os.makedirs(output_dir, exist_ok=True)
    results_dict = {}

    for func_class in cec_funcs:
        func_name = func_class.__name__
        print(f"Running {func_name} from {year} with dimension {dim}...")

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
                result = mod_woa_sa(func_obj.evaluate, lower, upper, evals, num_whales, dim)
                run_results.append(result)
            except Exception as e:
                print(f"Error during optimization of {func_name} (Run {i+1}): {e}")
                run_results.append(np.nan) # Append NaN if optimization fails

        results_dict[func_name] = run_results

    # Create DataFrame and save as CSV
    df = pd.DataFrame.from_dict(results_dict, orient='index',
                                 columns=[f'Run {i + 1}' for i in range(runs)])
    csv_path = os.path.join(output_dir, f'{year}_dim_{dim}_results.csv')
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")


# === Run CEC-2014, CEC-2017, CEC-2020, and CEC-2022 Benchmarks with appropriate dimensions ===
if __name__ == "__main__":
    common_dim = 30  # Using dimension 30 for CEC-2014, 2017, and 2020
    cec2022_dim = 20 # Using dimension 20 for CEC-2022

    run_cec_benchmark(opfunu.get_functions_based_classname("2014"), year="CEC-2014", dim=common_dim, output_dir="cec_2014_results")
    run_cec_benchmark(opfunu.get_functions_based_classname("2017"), year="CEC-2017", dim=common_dim, output_dir="cec_2017_results")
    run_cec_benchmark(opfunu.get_functions_based_classname("2020"), year="CEC-2020", dim=common_dim, output_dir="cec_2020_results")
    run_cec_benchmark(opfunu.get_functions_based_classname("2022"), year="CEC-2022", dim=cec2022_dim, output_dir="cec_2022_results")

    print("Benchmark runs completed.")