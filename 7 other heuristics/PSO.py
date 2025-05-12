import numpy as np
import os
import pandas as pd
import opfunu

# === Particle Swarm Optimization (PSO) ===
def particle_swarm_optimization(objective_function, lower_bound, upper_bound, max_evals, pop_size, dim):
    eval_counter = [0]  # Use mutable list for evaluation count

    def wrapped_func(x):
        if eval_counter[0] >= max_evals:
            raise StopIteration
        eval_counter[0] += 1
        return objective_function(x)

    # Initialize particles and velocities
    particles = np.random.uniform(lower_bound, upper_bound, (pop_size, dim))
    velocities = np.random.uniform(-0.1 * (upper_bound - lower_bound), 0.1 * (upper_bound - lower_bound), (pop_size, dim))  # Initialize velocities
    personal_best_positions = particles.copy()
    personal_best_fitnesses = np.array([wrapped_func(particle) for particle in particles])

    # Find initial global best particle
    global_best_index = np.argmin(personal_best_fitnesses)
    global_best_position = particles[global_best_index].copy()
    global_best_fitness = personal_best_fitnesses[global_best_index]

    # PSO parameters
    w = 0.7  # Inertia weight
    c1 = 1.4  # Cognitive coefficient
    c2 = 1.4  # Social coefficient

    try:
        while eval_counter[0] < max_evals:
            for i in range(pop_size):
                # Update velocity
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                velocities[i] = w * velocities[i] + \
                               c1 * r1 * (personal_best_positions[i] - particles[i]) + \
                               c2 * r2 * (global_best_position - particles[i])

                # Update particle position
                particles[i] = particles[i] + velocities[i]

                # Clip particles to the search space boundaries
                particles[i] = np.clip(particles[i], lower_bound, upper_bound)

                # Evaluate fitness
                fitness = wrapped_func(particles[i])

                # Update personal best
                if fitness < personal_best_fitnesses[i]:
                    personal_best_fitnesses[i] = fitness
                    personal_best_positions[i] = particles[i].copy()

                # Update global best
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particles[i].copy()

    except StopIteration:
        pass

    return global_best_fitness



# === Benchmark Runner for CEC Benchmarks (2014, 2017, 2020, and 2022) ===
def run_cec_benchmark_pso(cec_funcs, year="CEC-2020", dim=30, runs=50, evals=1200, pop_size=100, output_dir="cec_pso_results"):
    os.makedirs(output_dir, exist_ok=True)
    results_dict = {}

    for func_class in cec_funcs:
        func_name = func_class.__name__
        print(f"Running PSO on {func_name} from {year} with dimension {dim}...")

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
                result = particle_swarm_optimization(func_obj.evaluate, lower, upper, evals, pop_size, dim)
                run_results.append(result)
            except Exception as e:
                print(f"Error during optimization of {func_name} (Run {i+1}): {e}")
                run_results.append(np.nan)

        results_dict[func_name] = run_results

    # Create DataFrame and save as CSV
    df = pd.DataFrame.from_dict(results_dict, orient='index',
                                 columns=[f'Run {i + 1}' for i in range(runs)])
    csv_path = os.path.join(output_dir, f'{year}_dim_{dim}_pso_results.csv')
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")



# === Run CEC Benchmarks with PSO ===
if __name__ == "__main__":
    common_dim = 30
    cec2022_dim = 20
    pop_size = 100
    max_evaluations = 1200 * 5

    run_cec_benchmark_pso(opfunu.get_functions_based_classname("2014"), year="CEC-2014", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_pso_results")
    run_cec_benchmark_pso(opfunu.get_functions_based_classname("2017"), year="CEC-2017", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_pso_results")
    run_cec_benchmark_pso(opfunu.get_functions_based_classname("2020"), year="CEC-2020", dim=common_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_pso_results")
    run_cec_benchmark_pso(opfunu.get_functions_based_classname("2022"), year="CEC-2022", dim=cec2022_dim, runs=50, evals=max_evaluations, pop_size=pop_size, output_dir="cec_pso_results")

    print("PSO benchmark runs completed.")
