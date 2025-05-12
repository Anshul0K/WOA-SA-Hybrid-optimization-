import os
import pandas as pd
from scipy.stats import wilcoxon
from itertools import combinations

# === Configuration ===
folders = ["CEC-2014", "CEC-2017", "CEC-2020", "CEC-2022"]
filenames = ["WOA_SA.csv", "GA.csv", "PSO.csv", "GWO.csv", "DA.csv", "CHAO.csv", "BOA.csv", "FOA.csv"]

# === Function to perform Wilcoxon test for one folder ===
def process_folder(folder):
    print(f"\nProcessing: {folder}")
    data = {}
    
    # Load all available files
    for file in filenames:
        path = os.path.join(folder, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.set_index(df.columns[0], inplace=True)
            data[file] = df
        else:
            print(f"Missing file: {file} in {folder}, skipping it.")

    result_rows = []

    # Perform pairwise comparisons only for available files
    if not data:
        print(f"No data loaded for folder: {folder}")
        return

    for func_name in next(iter(data.values())).index:
        available_files = list(data.keys())
        for file1, file2 in combinations(available_files, 2):
            try:
                vals1 = data[file1].loc[func_name].values
                vals2 = data[file2].loc[func_name].values
                stat, p = wilcoxon(vals1, vals2)
                result_rows.append({
                    "Function": func_name,
                    "Algorithm 1": file1.replace(".csv", ""),
                    "Algorithm 2": file2.replace(".csv", ""),
                    "p-value": round(p, 5),
                    "Significant (p < 0.05)": "Yes" if p < 0.05 else "No"
                })
            except Exception as e:
                result_rows.append({
                    "Function": func_name,
                    "Algorithm 1": file1.replace(".csv", ""),
                    "Algorithm 2": file2.replace(".csv", ""),
                    "p-value": "Error",
                    "Significant (p < 0.05)": str(e)
                })

    result_df = pd.DataFrame(result_rows)
    output_path = os.path.join(folder, f"wilcoxon_results_{folder}.csv")
    result_df.to_csv(output_path, index=False)
    print(f"Saved Wilcoxon results to: {output_path}")

# === Run for all folders ===
if __name__ == "__main__":
    for folder in folders:
        process_folder(folder)
