import os
import pandas as pd
import numpy as np

# === Configuration ===
folders = ["CEC-2014", "CEC-2017", "CEC-2020", "CEC-2022"]
filenames = ["WOA_SA.csv", "GA.csv", "PSO.csv", "GWO.csv", "DA.csv", "CHAO.csv", "BOA.csv", "FOA.csv"]

def process_folder(folder):
    print(f"\nProcessing: {folder}")
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

    # Prepare result DataFrame
    function_names = next(iter(data.values())).index
    stats_rows = []

    for func in function_names:
        row = {"Function": func}
        means = {}
        stds = {}

        for algo, df in data.items():
            values = df.loc[func].values.astype(float)
            mean = np.mean(values)
            std = np.std(values)
            row[f"{algo}_Mean"] = round(mean, 6)
            row[f"{algo}_Std"] = round(std, 6)
            means[algo] = mean
            stds[algo] = std

        # Compute ranking (lower mean = better rank)
        sorted_algos = sorted(means, key=means.get)
        for rank, algo in enumerate(sorted_algos, start=1):
            row[f"{algo}_Rank"] = rank

        stats_rows.append(row)

    result_df = pd.DataFrame(stats_rows)
    output_path = os.path.join(folder, f"statistical_summary_{folder}.csv")
    result_df.to_csv(output_path, index=False)
    print(f"Saved statistical summary to: {output_path}")

# === Run for all folders ===
if __name__ == "__main__":
    for folder in folders:
        process_folder(folder)
