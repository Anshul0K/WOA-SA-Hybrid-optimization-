#WOA+PSO
import numpy as np
import os
import pandas as pd
import opfunu

# === Hybrid Whale Optimization Algorithm with Particle Swarm Optimization ===
def hybrid_woa_pso(objective_function, lower_bound, upper_bound, max_evals, num_agents, dim, w=0.729, c1=1.49445, c2=1.49445):
    eval_counter = [0]  # Use mutable list for evaluation count

    def wrapped_func(x):
        if eval_counter[0] >= max_evals:
            raise StopIteration
        eval_counter[0] += 1
        return objective_function(x)

    # Initialize agents (whales/particles)
    agents = np.random.uniform(lower_bound, upper_bound, (num_agents, dim))
    fitness = np.array([wrapped_func(agent) for agent in agents])

    # Initialize velocities (for PSO component)
    velocities = np.random.uniform(-abs(upper_bound - lower_bound) * 0.1, abs(upper_bound - lower_bound) * 0.1, (num_agents, dim))

    # Initialize personal best positions and fitness
    personal_best_positions = agents.copy()
    personal_best_fitness = fitness.copy()

    # Find initial global best
    global_best_index = np.argmin(fitness)
    global_best_position = agents[global_best_index].copy()
    global_best_fitness = fitness[global_best_index]

    try:
        while eval_counter[0] < max_evals:
            t = eval_counter[0] / max_evals
            a = 2 - 2 * t  # Control parameter for encircling prey
            a2 = -1 + 2 * t  # Control parameter for spiral updating
            b = 1  # Constant for defining the shape of the spiral
            l = np.random.uniform(-1, 1, (num_agents, 1))
            p = np.random.rand(num_agents)

            for i in range(num_agents):
                r1 = np.random.rand()
                r2 = np.random.rand()

                A = 2 * a * r1 - a
                C = 2 * r2

                if p[i] < 0.5:
                    if np.abs(A) >= 1:  # Exploration (random search)
                        rand_index = np.random.randint(num_agents)
                        X_rand = agents[rand_index]
                        D_xrand = np.abs(C * X_rand - agents[i])
                        agents[i] = X_rand - A * D_xrand
                    else:  # Exploitation (encircling prey or spiral updating)
                        D = np.abs(C * global_best_position - agents[i])
                        if np.random.rand() < 0.5:  # Encircling prey
                            agents[i] = global_best_position - A * D
                        else:  # Spiral updating
                            distance = np.abs(global_best_position - agents[i])
                            agents[i] = distance * np.exp(b * l[i]) * np.cos(2 * np.pi * l[i]) + global_best_position
                else:  # PSO influence
                    # Update velocity
                    velocities[i] = w * velocities[i] + \
                                    c1 * r1 * (personal_best_positions[i] - agents[i]) + \
                                    c2 * r2 * (global_best_position - agents[i])
                    # Update position
                    agents[i] = agents[i] + velocities[i]

                # Clip agents to the search space boundaries
                agents[i] = np.clip(agents[i], lower_bound, upper_bound)

                # Evaluate fitness of the new agent
                current_fitness = wrapped_func(agents[i])

                # Update personal best
                if current_fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = current_fitness
                    personal_best_positions[i] = agents[i].copy()

                # Update global best
                if current_fitness < global_best_fitness:
                    global_best_fitness = current_fitness
                    global_best_position = agents[i].copy()

    except StopIteration:
        pass

    return global_best_fitness


# === Benchmark Runner for CEC Benchmarks (2014, 2017, 2020, and 2022) ===
def run_cec_benchmark_hybrid(cec_funcs, year="CEC-2020", dim=30, runs=50, evals=1200, num_agents=30, output_dir="cec_hybrid_results"):
    os.makedirs(output_dir, exist_ok=True)
    results_dict = {}

    for func_class in cec_funcs:
        func_name = func_class.__name__
        print(f"Running Hybrid WOA-PSO on {func_name} from {year} with dimension {dim}...")

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
                result = hybrid_woa_pso(func_obj.evaluate, lower, upper, evals, num_agents, dim)
                run_results.append(result)
            except Exception as e:
                print(f"Error during optimization of {func_name} (Run {i+1}): {e}")
                run_results.append(np.nan)

        results_dict[func_name] = run_results

    # Create DataFrame and save as CSV
    df = pd.DataFrame.from_dict(results_dict, orient='index',
                                 columns=[f'Run {i + 1}' for i in range(runs)])
    csv_path = os.path.join(output_dir, f'{year}_dim_{dim}_hybrid_results.csv')
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")


# === Run CEC Benchmarks with Hybrid WOA-PSO ===
if __name__ == "__main__":
    common_dim = 30
    cec2022_dim = 20
    num_agents = 30  # Number of agents for the hybrid algorithm
    max_evaluations = 1200 * 5  # Adjust max evaluations if needed

    run_cec_benchmark_hybrid(opfunu.get_functions_based_classname("2014"), year="CEC-2014", dim=common_dim, runs=50, evals=max_evaluations, num_agents=num_agents, output_dir="cec_hybrid_results")
    run_cec_benchmark_hybrid(opfunu.get_functions_based_classname("2017"), year="CEC-2017", dim=common_dim, runs=50, evals=max_evaluations, num_agents=num_agents, output_dir="cec_hybrid_results")
    run_cec_benchmark_hybrid(opfunu.get_functions_based_classname("2020"), year="CEC-2020", dim=common_dim, runs=50, evals=max_evaluations, num_agents=num_agents, output_dir="cec_hybrid_results")
    run_cec_benchmark_hybrid(opfunu.get_functions_based_classname("2022"), year="CEC-2022", dim=cec2022_dim, runs=50, evals=max_evaluations, num_agents=num_agents, output_dir="cec_hybrid_results")

    print("Hybrid WOA-PSO benchmark runs completed.")