#WOA Crossover
import numpy as np
import os
import pandas as pd
import opfunu

# === Whale Optimization Algorithm with Crossover ===
def woa_with_crossover(objective_function, lower_bound, upper_bound, max_evals, num_whales, dim, crossover_rate=0.5):
    eval_counter = [0]  # Use mutable list for evaluation count

    def wrapped_func(x):
        if eval_counter[0] >= max_evals:
            raise StopIteration
        eval_counter[0] += 1
        return objective_function(x)

    # Initialize whales
    whales = np.random.uniform(lower_bound, upper_bound, (num_whales, dim))
    fitness = np.array([wrapped_func(whale) for whale in whales])

    # Find initial best whale
    best_index = np.argmin(fitness)
    best_whale = whales[best_index].copy()
    best_fitness = fitness[best_index]

    try:
        while eval_counter[0] < max_evals:
            t = eval_counter[0] / max_evals
            a = 2 - 2 * t  # Control parameter for encircling prey
            a2 = -1 + 2 * t  # Control parameter for spiral updating
            b = 1  # Constant for defining the shape of the spiral
            l = np.random.uniform(-1, 1, (num_whales, 1))
            p = np.random.rand(num_whales)

            # Apply standard WOA operators
            new_whales = np.zeros_like(whales)
            for i in range(num_whales):
                r1 = np.random.rand()
                r2 = np.random.rand()
                A = 2 * a * r1 - a
                C = 2 * r2

                if p[i] < 0.5:
                    if np.abs(A) >= 1:  # Exploration (random search)
                        rand_index = np.random.randint(num_whales)
                        X_rand = whales[rand_index]
                        D_xrand = np.abs(C * X_rand - whales[i])
                        new_whales[i] = X_rand - A * D_xrand
                    else:  # Exploitation (encircling prey or spiral updating)
                        D = np.abs(C * best_whale - whales[i])
                        if np.random.rand() < 0.5:  # Encircling prey
                            new_whales[i] = best_whale - A * D
                        else:  # Spiral updating
                            distance = np.abs(best_whale - whales[i])
                            new_whales[i] = distance * np.exp(b * l[i]) * np.cos(2 * np.pi * l[i]) + best_whale
                else:
                    new_whales[i] = whales[i].copy()  # No change for this whale initially

            # Apply crossover to a subset of the population
            for i in range(0, num_whales - 1, 2):
                if np.random.rand() < crossover_rate:
                    crossover_point = np.random.randint(1, dim)
                    # Create two offspring using single-point crossover
                    offspring1 = np.concatenate((new_whales[i][:crossover_point], new_whales[i + 1][crossover_point:]))
                    offspring2 = np.concatenate((new_whales[i + 1][:crossover_point], new_whales[i][crossover_point:]))
                    new_whales[i] = offspring1
                    new_whales[i + 1] = offspring2

            # Clip new whales to the search space boundaries
            new_whales = np.clip(new_whales, lower_bound, upper_bound)
            whales = new_whales
            fitness = np.array([wrapped_func(whale) for whale in whales])

            # Update the best whale
            current_best_index = np.argmin(fitness)
            if fitness[current_best_index] < best_fitness:
                best_fitness = fitness[current_best_index]
                best_whale = whales[current_best_index].copy()

    except StopIteration:
        pass

    return best_fitness


# === Benchmark Runner for CEC Benchmarks (2014, 2017, 2020, and 2022) ===
def run_cec_benchmark_crossover(cec_funcs, year="CEC-2020", dim=30, runs=50, evals=1200, num_whales=30, output_dir="cec_crossover_results"):
    os.makedirs(output_dir, exist_ok=True)
    results_dict = {}

    for func_class in cec_funcs:
        func_name = func_class.__name__
        print(f"Running WOA with Crossover on {func_name} from {year} with dimension {dim}...")

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
                result = woa_with_crossover(func_obj.evaluate, lower, upper, evals, num_whales, dim)
                run_results.append(result)
            except Exception as e:
                print(f"Error during optimization of {func_name} (Run {i+1}): {e}")
                run_results.append(np.nan)

        results_dict[func_name] = run_results

    # Create DataFrame and save as CSV
    df = pd.DataFrame.from_dict(results_dict, orient='index',
                                 columns=[f'Run {i + 1}' for i in range(runs)])
    csv_path = os.path.join(output_dir, f'{year}_dim_{dim}_crossover_woa_results.csv')
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")


# === Run CEC Benchmarks with WOA and Crossover ===
if __name__ == "__main__":
    common_dim = 30
    cec2022_dim = 20
    num_whales = 30
    max_evaluations = 1200 * 5
    crossover_rate = 0.7  # Adjust the crossover rate

    run_cec_benchmark_crossover(opfunu.get_functions_based_classname("2014"), year="CEC-2014", dim=common_dim, runs=50, evals=max_evaluations, num_whales=num_whales, output_dir="cec_crossover_results")
    run_cec_benchmark_crossover(opfunu.get_functions_based_classname("2017"), year="CEC-2017", dim=common_dim, runs=50, evals=max_evaluations, num_whales=num_whales, output_dir="cec_crossover_results")
    run_cec_benchmark_crossover(opfunu.get_functions_based_classname("2020"), year="CEC-2020", dim=common_dim, runs=50, evals=max_evaluations, num_whales=num_whales, output_dir="cec_crossover_results")
    run_cec_benchmark_crossover(opfunu.get_functions_based_classname("2022"), year="CEC-2022", dim=cec2022_dim, runs=50, evals=max_evaluations, num_whales=num_whales, output_dir="cec_crossover_results")

    print("WOA with Crossover benchmark runs completed.")