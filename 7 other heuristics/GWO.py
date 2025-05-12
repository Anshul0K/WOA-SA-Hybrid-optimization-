import numpy as np
import os
import pandas as pd
import opfunu

# === Grey Wolf Optimizer (GWO) ===
def grey_wolf_optimization(objective_function, lower_bound, upper_bound, max_evals, pop_size, dim):
    eval_counter = [0]  # Use mutable list for evaluation count

    def wrapped_func(x):
        if eval_counter[0] >= max_evals:
            raise StopIteration
        eval_counter[0] += 1
        return objective_function(x)

    # Initialize wolves
    wolves = np.random.uniform(lower_bound, upper_bound, (pop_size, dim))
    fitness = np.array([wrapped_func(wolf) for wolf in wolves])

    # Initialize alpha, beta, and delta wolf positions and their fitness
    alpha_index = np.argmin(fitness)
    alpha_position = wolves[alpha_index].copy()
    alpha_fitness = fitness[alpha_index]

    beta_index = np.argsort(fitness)[1]  # Second best
    beta_position = wolves[beta_index].copy()
    beta_fitness = fitness[beta_index]

    delta_index = np.argsort(fitness)[2]  # Third best
    delta_position = wolves[delta_index].copy()
    delta_fitness = fitness[delta_index]

    t = 0  # Iteration counter

    try:
        while eval_counter[0] < max_evals:
            t += 1
            a = 2 - t * (2 / max_evals)  # a decreases linearly from 2 to 0

            for i in range(pop_size):
                # Calculate coefficients D and C
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)

                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                r3 = np.random.rand(dim)
                r4 = np.random.rand(dim)

                A2 = 2 * a * r3 - a
                C2 = 2 * r4

                r5 = np.random.rand(dim)
                r6 = np.random.rand(dim)

                A3 = 2 * a * r5 - a
                C3 = 2 * r6

                # Calculate distances to alpha, beta, and delta
                D_alpha = np.abs(C1 * alpha_position - wolves[i])
                D_beta = np.abs(C2 * beta_position - wolves[i])
                D_delta = np.abs(C3 * delta_position - wolves[i])

                # Calculate X1, X2, and X3
                X1 = alpha_position - A1 * D_alpha
                X2 = beta_position - A2 * D_beta
                X3 = delta_position - A3 * D_delta

                # Update wolf position
                wolves[i] = (X1 + X2 + X3) / 3

                # Clip wolves to the search space boundaries
                wolves[i] = np.clip(wolves[i], lower_bound, upper_bound)

                # Evaluate fitness
                fitness[i] = wrapped_func(wolves[i])

                # Update alpha, beta, and delta wolves
                if fitness[i] < alpha_fitness:
                    alpha_fitness = fitness[i]
                    alpha_position = wolves[i].copy()

                    beta_fitness = alpha_fitness # make beta = alpha before updating it
                    beta_position = alpha_position.copy()

                    delta_fitness = beta_fitness # make delta = beta before updating it
                    delta_position = beta_position.copy()

                elif fitness[i] < beta_fitness:
                    beta_fitness = fitness[i]
                    beta_position = wolves[i].copy()

                    delta_fitness = beta_fitness # make delta = beta
                    delta_position = beta_position.copy()

                elif fitness[i] < delta_fitness:
                    delta_fitness = fitness[i]
                    delta_position = wolves[i].copy()

    except StopIteration:
        pass

    return alpha_fitness  # Return the fitness of the alpha wolf (best solution)



# === Benchmark Runner for CEC Benchmarks (2014, 2017, 2020, and 2022) ===
def run_cec_benchmark_gwo(cec_funcs, year="CEC-2020", dim=30, runs=50, evals=1200, pop_size=100, output_dir="cec_gwo_results"):
    os.makedirs(output_dir, exist_ok=True)
    results_dict = {}

    for func_class in cec_funcs:
        func_name = func_class.__name__
        print(f"Running GWO on {func_name} from {year} with dimension {dim}...")

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
                result = grey_wolf_optimization(func_obj.evaluate, lower, upper, evals, pop_size, dim)
                run_results.append(result)
            except Exception as e:
                print(f"Error during optimization of {func_name} (Run {i+1}): {e}")
                run_results.append(np.nan)

        results_dict[func_name] = run_results

    # Create DataFrame and save as CSV
    df = pd.DataFrame.from_dict(results_dict, orient='index',
                                 columns=[f'Run {i + 1}' for i in range(runs)])
    csv_path = os.path.join(output_dir, f'{year}_dim_{dim}_gwo_results.csv')
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")



# === Run CEC Benchmarks with GWO ===
if __name__ == "__main__":
    common_dim = 30
    cec2022_dim = 20
    pop_size = 100
    max_evaluations = 1200 * 5

    run_cec_benchmark_gwo(opfunu.get_functions_based_classname("2014"), year="CEC-2014", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_gwo_results")
    run_cec_benchmark_gwo(opfunu.get_functions_based_classname("2017"), year="CEC-2017", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_gwo_results")
    run_cec_benchmark_gwo(opfunu.get_functions_based_classname("2020"), year="CEC-2020", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_gwo_results")
    run_cec_benchmark_gwo(opfunu.get_functions_based_classname("2022"), year="CEC-2022", dim=cec2022_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_gwo_results")

    print("GWO benchmark runs completed.")
