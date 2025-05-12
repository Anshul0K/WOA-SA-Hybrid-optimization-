import numpy as np
import os
import pandas as pd
import opfunu

# === Fox Optimization Algorithm (FOA) ===
def fox_optimization_algorithm(objective_function, lower_bound, upper_bound, max_evals, pop_size, dim):
    eval_counter = [0]  # Use mutable list for evaluation count

    def wrapped_func(x):
        if eval_counter[0] >= max_evals:
            raise StopIteration
        eval_counter[0] += 1
        return objective_function(x)

    # Initialize foxes
    foxes = np.random.uniform(lower_bound, upper_bound, (pop_size, dim))
    fitness = np.array([wrapped_func(fox) for fox in foxes])

    # Find initial best fox (prey)
    best_index = np.argmin(fitness)
    prey_position = foxes[best_index].copy()
    prey_fitness = fitness[best_index]

    # FOA parameters
    Q_max = 10  # Maximum iteration for group attacking
    visual_distance = (upper_bound - lower_bound) / 2  # Simplified visual distance.  Can be adapted.
    G = 1 # Group attacking number
    t = 0

    try:
        while eval_counter[0] < max_evals:
            t += 1
            for i in range(pop_size):
                if np.random.rand() < 0.5: # Individual hunting
                    # Fox moves towards a random prey
                    rand_fox_index = np.random.randint(pop_size)
                    distance = np.abs(foxes[i] - foxes[rand_fox_index])
                    foxes[i] = foxes[i] + np.random.rand() * distance
                else: # Group hunting
                    if t < Q_max:
                         distance = np.abs(prey_position - foxes[i])
                         foxes[i] = prey_position + np.random.rand() * distance
                    else:
                        #position update
                        distance = np.abs(prey_position - foxes[i])
                        foxes[i] = prey_position + np.random.rand() * distance
                # Clip foxes to the search space boundaries
                foxes[i] = np.clip(foxes[i], lower_bound, upper_bound)

                # Evaluate fitness
                fitness[i] = wrapped_func(foxes[i])

            # Update the best fox (prey)
            current_best_index = np.argmin(fitness)
            if fitness[current_best_index] < prey_fitness:
                prey_fitness = fitness[current_best_index]
                prey_position = foxes[current_best_index].copy()

    except StopIteration:
        pass

    return prey_fitness



# === Benchmark Runner for CEC Benchmarks (2014, 2017, 2020, and 2022) ===
def run_cec_benchmark_foa(cec_funcs, year="CEC-2020", dim=30, runs=50, evals=1200, pop_size=100, output_dir="cec_foa_results"):
    os.makedirs(output_dir, exist_ok=True)
    results_dict = {}

    for func_class in cec_funcs:
        func_name = func_class.__name__
        print(f"Running FOA on {func_name} from {year} with dimension {dim}...")

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
                result = fox_optimization_algorithm(func_obj.evaluate, lower, upper, evals, pop_size, dim)
                run_results.append(result)
            except Exception as e:
                print(f"Error during optimization of {func_name} (Run {i+1}): {e}")
                run_results.append(np.nan)

        results_dict[func_name] = run_results

    # Create DataFrame and save as CSV
    df = pd.DataFrame.from_dict(results_dict, orient='index',
                                 columns=[f'Run {i + 1}' for i in range(runs)])
    csv_path = os.path.join(output_dir, f'{year}_dim_{dim}_foa_results.csv')
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")



# === Run CEC Benchmarks with FOA ===
if __name__ == "__main__":
    common_dim = 30
    cec2022_dim = 20
    pop_size = 100
    max_evaluations = 1200 * 5

    run_cec_benchmark_foa(opfunu.get_functions_based_classname("2014"), year="CEC-2014", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_foa_results")
    run_cec_benchmark_foa(opfunu.get_functions_based_classname("2017"), year="CEC-2017", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_foa_results")
    run_cec_benchmark_foa(opfunu.get_functions_based_classname("2020"), year="CEC-2020", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_foa_results")
    run_cec_benchmark_foa(opfunu.get_functions_based_classname("2022"), year="CEC-2022", dim=cec2022_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_foa_results")

    print("FOA benchmark runs completed.")
