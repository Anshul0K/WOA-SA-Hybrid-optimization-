import os
import pandas as pd
import matplotlib.pyplot as plt

# === Configuration ===
folders = ["CEC-2014", "CEC-2017", "CEC-2020", "CEC-2022"]
filenames = ["WOA_SA.csv", "WOA_GA.csv", "WOA_PSO.csv", "WOA_Cros.csv", "WOA_Mut.csv", "WOA_original.csv"]

# === Function to generate and save convergence plots for each folder ===
def plot_convergence(folder):
    print(f"\nGenerating convergence curves for {folder}")
    data = {}

    # Load available CSVs
    for file in filenames:
        path = os.path.join(folder, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.set_index(df.columns[0], inplace=True)
            data[file.replace(".csv", "")] = df
        else:
            print(f"Missing file: {file} in {folder}, skipping it.")

    if not data:
        print(f"No valid data found in {folder}, skipping.")
        return

    # Create a plot for each function
    function_names = next(iter(data.values())).index

    for func_name in function_names:
        plt.figure(figsize=(10, 6))  # Create a new figure for each function

        # Plot convergence curves for each algorithm
        for algo, df in data.items():
            fitness_values = df.loc[func_name].values.astype(float)  # Extract the fitness values
            plt.plot(range(1, len(fitness_values) + 1), fitness_values, label=algo)

        plt.title(f"Convergence Curves for {func_name} ({folder})")
        plt.xlabel("Iterations")
        plt.ylabel("Objective Value")
        plt.legend(title="Algorithms")
        plt.grid(True)

        # Save the plot
        plot_path = os.path.join(folder, f"convergence_{func_name}_{folder}.png")
        plt.savefig(plot_path)
        plt.close()  # Close the plot to free memory
        print(f"Saved convergence plot for {func_name} to: {plot_path}")

# === Run for all folders ===
if __name__ == "__main__":
    for folder in folders:
        plot_convergence(folder)
