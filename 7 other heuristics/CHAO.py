import numpy as np
import os
import pandas as pd
import opfunu

# === Chimp Optimization Algorithm (ChOA) ===
def chimp_optimization_algorithm(objective_function, lower_bound, upper_bound, max_evals, pop_size, dim):
    eval_counter = [0]  # Use mutable list for evaluation count

    def wrapped_func(x):
        if eval_counter[0] >= max_evals:
            raise StopIteration
        eval_counter[0] += 1
        return objective_function(x)

    # Initialize chimps
    chimps = np.random.uniform(lower_bound, upper_bound, (pop_size, dim))
    velocities = np.random.uniform(-0.1 * (upper_bound - lower_bound), 0.1 * (upper_bound - lower_bound), (pop_size, dim))  # Initialize velocities
    fitness = np.array([wrapped_func(chimp) for chimp in chimps])

    # Find initial best chimp (prey) and attacker
    best_index = np.argmin(fitness)
    prey_position = chimps[best_index].copy()
    prey_fitness = fitness[best_index]

    attacker_index = np.argmax(fitness)  # Simplification:  Worst is attacker.  More complex versions exist.
    attacker_position = chimps[attacker_index].copy() #initial attacker position
    attacker_fitness = fitness[attacker_index]

    # ChOA parameters
    alpha = 2       # attack power
    beta = 2        # chaos factor
    mu = 0.1      # chimp group influence
    w = 0.7       # Inertia weight
    t = 0

    try:
        while eval_counter[0] < max_evals:
            t += 1
            for i in range(pop_size):
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()
                r4 = np.random.rand()

                a = 2 * r1 - alpha
                b = 2 * r2 - beta

                # Chimp group behavior
                if np.random.rand() < 0.5:
                    d_prey = np.abs(prey_position - chimps[i])
                    x1 = prey_position - a * d_prey
                    if np.random.rand() < 0.5:
                        d_attacker = np.abs(attacker_position - chimps[i])
                        x2 = attacker_position - a * d_attacker
                    else:
                        x2 = chimps[i] - np.random.uniform(-1, 1, dim) * beta
                    chimps[i] = (x1 + x2) / 2
                else:
                    chimps[i] = prey_position - np.random.uniform(-1, 1, dim) * mu


                # Clip chimps to the search space boundaries
                chimps[i] = np.clip(chimps[i], lower_bound, upper_bound)

                # Evaluate fitness
                fitness[i] = wrapped_func(chimps[i])

                # Update prey (best solution)
                if fitness[i] < prey_fitness:
                    prey_fitness = fitness[i]
                    prey_position = chimps[i].copy()

                # Update attacker (worst solution) - simplified
                if fitness[i] > attacker_fitness:
                    attacker_fitness = fitness[i]
                    attacker_position = chimps[i].copy()

    except StopIteration:
        pass

    return prey_fitness



# === Benchmark Runner for CEC Benchmarks (2014, 2017, 2020, and 2022) ===
def run_cec_benchmark_choa(cec_funcs, year="CEC-2020", dim=30, runs=50, evals=1200, pop_size=100, output_dir="cec_choa_results"):
    os.makedirs(output_dir, exist_ok=True)
    results_dict = {}

    for func_class in cec_funcs:
        func_name = func_class.__name__
        print(f"Running ChOA on {func_name} from {year} with dimension {dim}...")

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
                result = chimp_optimization_algorithm(func_obj.evaluate, lower, upper, evals, pop_size, dim)
                run_results.append(result)
            except Exception as e:
                print(f"Error during optimization of {func_name} (Run {i+1}): {e}")
                run_results.append(np.nan)

        results_dict[func_name] = run_results

    # Create DataFrame and save as CSV
    df = pd.DataFrame.from_dict(results_dict, orient='index',
                                 columns=[f'Run {i + 1}' for i in range(runs)])
    csv_path = os.path.join(output_dir, f'{year}_dim_{dim}_choa_results.csv')
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")



# === Run CEC Benchmarks with ChOA ===
if __name__ == "__main__":
    common_dim = 30
    cec2022_dim = 20
    pop_size = 100
    max_evaluations = 1200 * 5

    run_cec_benchmark_choa(opfunu.get_functions_based_classname("2014"), year="CEC-2014", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_choa_results")
    run_cec_benchmark_choa(opfunu.get_functions_based_classname("2017"), year="CEC-2017", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_choa_results")
    run_cec_benchmark_choa(opfunu.get_functions_based_classname("2020"), year="CEC-2020", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_choa_results")
    run_cec_benchmark_choa(opfunu.get_functions_based_classname("2022"), year="CEC-2022", dim=cec2022_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_choa_results")

    print("ChOA benchmark runs completed.")
