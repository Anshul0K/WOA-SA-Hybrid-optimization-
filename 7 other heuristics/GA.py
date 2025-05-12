import numpy as np
import os
import pandas as pd
import opfunu

# === Genetic Algorithm (GA) ===
def genetic_algorithm(objective_function, lower_bound, upper_bound, max_evals, pop_size, dim, mutation_rate=0.01):
    eval_counter = [0]  # Use mutable list for evaluation count

    def wrapped_func(x):
        if eval_counter[0] >= max_evals:
            raise StopIteration
        eval_counter[0] += 1
        return objective_function(x)

    # Initialize population
    population = np.random.uniform(lower_bound, upper_bound, (pop_size, dim))
    fitness = np.array([wrapped_func(individual) for individual in population])

    # Find initial best individual
    best_index = np.argmin(fitness)
    best_individual = population[best_index].copy()
    best_fitness = fitness[best_index]

    try:
        while eval_counter[0] < max_evals:
            # Selection (Tournament Selection)
            tournament_size = 3
            selected_indices = []
            for _ in range(pop_size):
                tournament_candidates = np.random.choice(pop_size, tournament_size, replace=False)
                tournament_fitnesses = fitness[tournament_candidates]
                winner_index = tournament_candidates[np.argmin(tournament_fitnesses)]
                selected_indices.append(winner_index)
            selected_population = population[selected_indices]

            # Crossover (Single-Point Crossover)
            offspring = []
            for i in range(0, pop_size, 2):
                parent1 = selected_population[i % pop_size]
                parent2 = selected_population[(i + 1) % pop_size]
                crossover_point = np.random.randint(1, dim)
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                offspring.extend([child1, child2])
            offspring = np.array(offspring)

            # Mutation
            for i in range(pop_size):
                for j in range(dim):
                    if np.random.rand() < mutation_rate:
                        offspring[i, j] = np.random.uniform(lower_bound, upper_bound)

            # Evaluate offspring fitness
            offspring_fitness = np.array([wrapped_func(individual) for individual in offspring])

            # Replacement (Elitism + Replacement)
            # Keep the best individual
            new_population = np.zeros_like(population)
            new_population[0] = best_individual
            new_population[1:] = offspring[:-1] # Keep all offspring except the last one.  Important
            population = new_population
            fitness = np.array([wrapped_func(individual) for individual in population])


            # Update the best individual
            current_best_index = np.argmin(fitness)
            if fitness[current_best_index] < best_fitness:
                best_fitness = fitness[current_best_index]
                best_individual = population[current_best_index].copy()

    except StopIteration:
        pass
    return best_fitness



# === Benchmark Runner for CEC Benchmarks (2014, 2017, 2020, and 2022) ===
def run_cec_benchmark_ga(cec_funcs, year="CEC-2020", dim=30, runs=50, evals=1200, pop_size=100, mutation_rate=0.01, output_dir="cec_ga_results"):
    os.makedirs(output_dir, exist_ok=True)
    results_dict = {}

    for func_class in cec_funcs:
        func_name = func_class.__name__
        print(f"Running GA on {func_name} from {year} with dimension {dim}...")

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
                result = genetic_algorithm(func_obj.evaluate, lower, upper, evals, pop_size, dim, mutation_rate)
                run_results.append(result)
            except Exception as e:
                print(f"Error during optimization of {func_name} (Run {i+1}): {e}")
                run_results.append(np.nan)

        results_dict[func_name] = run_results

    # Create DataFrame and save as CSV
    df = pd.DataFrame.from_dict(results_dict, orient='index',
                                 columns=[f'Run {i + 1}' for i in range(runs)])
    csv_path = os.path.join(output_dir, f'{year}_dim_{dim}_ga_results.csv')
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")



# === Run CEC Benchmarks with GA ===
if __name__ == "__main__":
    common_dim = 30
    cec2022_dim = 20
    pop_size = 100
    max_evaluations = 1200 * 5
    mutation_rate = 0.01

    run_cec_benchmark_ga(opfunu.get_functions_based_classname("2014"), year="CEC-2014", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, mutation_rate=mutation_rate, output_dir="cec_ga_results")
    run_cec_benchmark_ga(opfunu.get_functions_based_classname("2017"), year="CEC-2017", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, mutation_rate=mutation_rate, output_dir="cec_ga_results")
    run_cec_benchmark_ga(opfunu.get_functions_based_classname("2020"), year="CEC-2020", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, mutation_rate=mutation_rate, output_dir="cec_ga_results")
    run_cec_benchmark_ga(opfunu.get_functions_based_classname("2022"), year="CEC-2022", dim=cec2022_dim, runs=50, evals=max_evaluations, pop_size=pop_size, mutation_rate=mutation_rate, output_dir="cec_ga_results")

    print("GA benchmark runs completed.")
