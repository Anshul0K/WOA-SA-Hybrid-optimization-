#WOA+GA
import numpy as np
import os
import pandas as pd
import opfunu

# === Hybrid Whale Optimization Algorithm with Genetic Algorithm ===
def hybrid_woa_ga(objective_function, lower_bound, upper_bound, max_evals, num_whales, dim,
                  population_size_ga=50, crossover_rate=0.8, mutation_rate=0.01, tournament_size=3):
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

            # Generate a subset of whales for GA operations
            num_ga_candidates = int(0.5 * num_whales)  # Apply GA to half the population
            ga_indices = np.random.choice(num_whales, size=num_ga_candidates, replace=False)
            ga_population = whales[ga_indices].copy()
            ga_fitness = fitness[ga_indices].copy()

            # --- Genetic Algorithm Operations ---
            if len(ga_population) >= 2:  # Ensure at least two individuals for crossover
                # Selection (Tournament Selection)
                def selection(population, fitness, tournament_size):
                    participants_indices = np.random.choice(len(population), tournament_size, replace=False)
                    participants = population[participants_indices]
                    participants_fitness = fitness[participants_indices]
                    winner_index = np.argmin(participants_fitness)
                    return population[participants_indices[winner_index]].copy()

                parents = [selection(ga_population, ga_fitness, tournament_size) for _ in range(len(ga_population))]
                parents = np.array(parents)

                # Crossover (Single-Point Crossover)
                offspring = []
                for i in range(0, len(parents), 2):
                    if np.random.rand() < crossover_rate and i + 1 < len(parents):
                        crossover_point = np.random.randint(1, dim)
                        child1 = np.concatenate((parents[i][:crossover_point], parents[i + 1][crossover_point:]))
                        child2 = np.concatenate((parents[i + 1][:crossover_point], parents[i][crossover_point:]))
                        offspring.extend([child1, child2])
                    else:
                        offspring.extend([parents[i].copy(), parents[i + 1].copy() if i + 1 < len(parents) else parents[i].copy()])
                offspring = np.array(offspring)

                # Mutation
                for i in range(len(offspring)):
                    if np.random.rand() < mutation_rate:
                        mutation_dim = np.random.randint(dim)
                        mutation_amount = np.random.uniform(-0.1 * (upper_bound - lower_bound), 0.1 * (upper_bound - lower_bound))
                        offspring[i, mutation_dim] = np.clip(offspring[i, mutation_dim] + mutation_amount, lower_bound, upper_bound)

                # Evaluate offspring fitness and replace part of the whale population
                offspring_fitness = np.array([wrapped_func(off) for off in offspring])

                # Replace the GA candidates with the new offspring if they are better
                for i in range(len(ga_indices)):
                    if offspring_fitness[i] < fitness[ga_indices[i]]:
                        whales[ga_indices[i]] = offspring[i].copy()
                        fitness[ga_indices[i]] = offspring_fitness[i]

            # --- Whale Optimization Algorithm Operators ---
            for i in range(num_whales):
                if i not in ga_indices:  # Apply WOA to the remaining whales
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A = 2 * a * r1 - a
                    C = 2 * r2

                    if p[i] < 0.5:
                        if np.abs(A) >= 1:  # Exploration (random search)
                            rand_index = np.random.randint(num_whales)
                            X_rand = whales[rand_index]
                            D_xrand = np.abs(C * X_rand - whales[i])
                            whales[i] = X_rand - A * D_xrand
                        else:  # Exploitation (encircling prey or spiral updating)
                            D = np.abs(C * best_whale - whales[i])
                            if np.random.rand() < 0.5:  # Encircling prey
                                whales[i] = best_whale - A * D
                            else:  # Spiral updating
                                distance = np.abs(best_whale - whales[i])
                                whales[i] = distance * np.exp(b * l[i]) * np.cos(2 * np.pi * l[i]) + best_whale
                    else:
                        # No direct PSO-like update here, focusing on WOA and GA
                        pass # Whales not undergoing GA use standard WOA update

                # Clip whales to the search space boundaries
                whales[i] = np.clip(whales[i], lower_bound, upper_bound)

                # Evaluate fitness (if not already done by GA)
                if i not in ga_indices:
                    fitness[i] = wrapped_func(whales[i])

            # Update the best whale
            current_best_index = np.argmin(fitness)
            if fitness[current_best_index] < best_fitness:
                best_fitness = fitness[current_best_index]
                best_whale = whales[current_best_index].copy()

    except StopIteration:
        pass

    return best_fitness


# === Benchmark Runner for CEC Benchmarks (2014, 2017, 2020, and 2022) ===
def run_cec_benchmark_ga_hybrid(cec_funcs, year="CEC-2020", dim=30, runs=50, evals=1200, num_whales=30, output_dir="cec_ga_hybrid_results"):
    os.makedirs(output_dir, exist_ok=True)
    results_dict = {}

    for func_class in cec_funcs:
        func_name = func_class.__name__
        print(f"Running Hybrid WOA-GA on {func_name} from {year} with dimension {dim}...")

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
                result = hybrid_woa_ga(func_obj.evaluate, lower, upper, evals, num_whales, dim)
                run_results.append(result)
            except Exception as e:
                print(f"Error during optimization of {func_name} (Run {i+1}): {e}")
                run_results.append(np.nan)

        results_dict[func_name] = run_results

    # Create DataFrame and save as CSV
    df = pd.DataFrame.from_dict(results_dict, orient='index',
                                 columns=[f'Run {i + 1}' for i in range(runs)])
    csv_path = os.path.join(output_dir, f'{year}_dim_{dim}_ga_hybrid_results.csv')
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")


# === Run CEC Benchmarks with Hybrid WOA-GA ===
if __name__ == "__main__":
    common_dim = 30
    cec2022_dim = 20
    num_whales = 30
    max_evaluations = 1200 * 5

    run_cec_benchmark_ga_hybrid(opfunu.get_functions_based_classname("2014"), year="CEC-2014", dim=common_dim, runs=50, evals=max_evaluations, num_whales=num_whales, output_dir="cec_ga_hybrid_results")
    run_cec_benchmark_ga_hybrid(opfunu.get_functions_based_classname("2017"), year="CEC-2017", dim=common_dim, runs=50, evals=max_evaluations, num_whales=num_whales, output_dir="cec_ga_hybrid_results")
    run_cec_benchmark_ga_hybrid(opfunu.get_functions_based_classname("2020"), year="CEC-2020", dim=common_dim, runs=50, evals=max_evaluations, num_whales=num_whales, output_dir="cec_ga_hybrid_results")
    run_cec_benchmark_ga_hybrid(opfunu.get_functions_based_classname("2022"), year="CEC-2022", dim=cec2022_dim, runs=50, evals=max_evaluations, num_whales=num_whales, output_dir="cec_ga_hybrid_results")

    print("Hybrid WOA-GA benchmark runs completed.")