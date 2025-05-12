import numpy as np
import os
import pandas as pd
import opfunu

# === Butterfly Optimization Algorithm (BOA) ===
def butterfly_optimization_algorithm(objective_function, lower_bound, upper_bound, max_evals, pop_size, dim):
    eval_counter = [0]  # Use mutable list for evaluation count

    def wrapped_func(x):
        if eval_counter[0] >= max_evals:
            raise StopIteration
        eval_counter[0] += 1
        return objective_function(x)

    # Initialize butterflies
    butterflies = np.random.uniform(lower_bound, upper_bound, (pop_size, dim))
    fitness = np.array([wrapped_func(butterfly) for butterfly in butterflies])

    # Find initial best butterfly
    best_index = np.argmin(fitness)
    best_position = butterflies[best_index].copy()
    best_fitness = fitness[best_index]

    # BOA parameters
    a = 0.1  # Power exponent
    c = 0.01 # sensory modality
    t = 0  # Iteration counter

    try:
        while eval_counter[0] < max_evals:
            t += 1
            for i in range(pop_size):
                # Calculate fragrance
                if np.random.rand() < 0.8: # local search
                    for k in range(dim):
                        delta = np.random.rand() * np.random.rand()
                        butterflies[i,k] = butterflies[i,k] + delta * (butterflies[i,k] - butterflies[np.random.randint(0,pop_size),k])
                else: # global search
                    for k in range(dim):
                         delta = np.random.rand() * np.random.rand()
                         butterflies[i,k] = best_position[k] + delta * (butterflies[i,k] - best_position[k])

                # Clip butterflies to the search space boundaries
                butterflies[i] = np.clip(butterflies[i], lower_bound, upper_bound)

                # Evaluate fitness
                fitness[i] = wrapped_func(butterflies[i])

            # Update the best butterfly
            current_best_index = np.argmin(fitness)
            if fitness[current_best_index] < best_fitness:
                best_fitness = fitness[current_best_index]
                best_position = butterflies[current_best_index].copy()

    except StopIteration:
        pass

    return best_fitness



# === Benchmark Runner for CEC Benchmarks (2014, 2017, 2020, and 2022) ===
def run_cec_benchmark_bao(cec_funcs, year="CEC-2020", dim=30, runs=50, evals=1200, pop_size=100, output_dir="cec_bao_results"):
    os.makedirs(output_dir, exist_ok=True)
    results_dict = {}

    for func_class in cec_funcs:
        func_name = func_class.__name__
        print(f"Running BOA on {func_name} from {year} with dimension {dim}...")

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
                result = butterfly_optimization_algorithm(func_obj.evaluate, lower, upper, evals, pop_size, dim)
                run_results.append(result)
            except Exception as e:
                print(f"Error during optimization of {func_name} (Run {i+1}): {e}")
                run_results.append(np.nan)

        results_dict[func_name] = run_results

    # Create DataFrame and save as CSV
    df = pd.DataFrame.from_dict(results_dict, orient='index',
                                 columns=[f'Run {i + 1}' for i in range(runs)])
    csv_path = os.path.join(output_dir, f'{year}_dim_{dim}_bao_results.csv')
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")



# === Run CEC Benchmarks with BOA ===
if __name__ == "__main__":
    common_dim = 30
    cec2022_dim = 20
    pop_size = 100
    max_evaluations = 1200 * 5

    run_cec_benchmark_bao(opfunu.get_functions_based_classname("2014"), year="CEC-2014", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_bao_results")
    run_cec_benchmark_bao(opfunu.get_functions_based_classname("2017"), year="CEC-2017", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_bao_results")
    run_cec_benchmark_bao(opfunu.get_functions_based_classname("2020"), year="CEC-2020", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_bao_results")
    run_cec_benchmark_bao(opfunu.get_functions_based_classname("2022"), year="CEC-2022", dim=cec2022_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_bao_results")

    print("BOA benchmark runs completed.")
