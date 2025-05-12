Enhanced Whale Optimization Algorithm using Simulated Annealing

Abstract

This report presents a comparative study of a modified Whale Optimization Algorithm (WOA) enhanced with Simulated Annealing (SA). The hybrid WOA+SA algorithm is benchmarked against five other WOA variants—WOA+PSO, WOA+GA, WOA+Mutation, WOA+Crossover, and original WOA—across standard CEC benchmark functions from 2014, 2017, 2020, and 2022. Each algorithm is evaluated over 50 iterations per function. Convergence curves, Wilcoxon signed-rank test, and statistical analysis are employed to assess and compare performance.

1. Introduction

Metaheuristic algorithms are widely used for solving complex optimization problems. Among them, the Whale Optimization Algorithm (WOA) has gained popularity due to its simplicity and effectiveness. However, it sometimes suffers from premature convergence and local optima entrapment. To address this, hybridization strategies are employed.

In this work, we hybridize WOA with Simulated Annealing (WOA+SA), aiming to improve its exploitation capabilities without compromising exploration. SA helps in probabilistic hill-climbing and allows escape from local optima, complementing the exploration mechanisms of WOA. This synergy motivated the choice of WOA+SA for enhancement.

2. Motivation for WOA+SA Hybrid

The main motivation behind combining Simulated Annealing with WOA stems from the following observations:

WOA Limitation: While WOA is efficient in global search, it may stagnate during the exploitation phase.

SA Strength: Simulated Annealing introduces a probabilistic acceptance mechanism that occasionally accepts worse solutions, enabling better local search and escape from local minima.

Hybrid Advantage: By integrating SA into the exploitation stage of WOA, we aim to balance exploration and exploitation more effectively, providing better convergence on complex landscapes.

WOA+SA stands out from other variants (e.g., WOA+PSO, WOA+GA) by not relying on swarm-level or evolutionary population mechanisms for improvement, but rather enhancing individual search agents' performance using thermodynamic principles.

3. Benchmark Setup

Benchmark Suites: CEC 2014, 2017, 2020, and 2022 benchmark functions.

Dimensionality: 30-Dimensional problems.

Runs: Each function was evaluated with 50 independent runs.

Algorithms:

WOA+SA (Proposed)

WOA+PSO

WOA+GA

WOA+Mutation

WOA+Crossover

WOA (Original)

4. Experimental Methodology

Each algorithm is tested on all benchmark functions.

The best score from each run is recorded.

Convergence behavior is captured per function.

Statistical summaries include mean, standard deviation, and ranking.

Wilcoxon signed-rank test is applied to assess significant differences.

5. Convergence Analysis

Convergence curves were plotted for each benchmark function to observe the optimization path for all six variants. WOA+SA consistently shows a smooth and deeper convergence compared to others.

6. Statistical Analysis

Each algorithm's performance is summarized by calculating:

Mean and standard deviation of scores for each function.

Rankings based on average score.

WOA+SA exhibited top ranks across most functions in all CEC sets.

7. Wilcoxon Signed-Rank Test

Pairwise Wilcoxon tests were performed between all algorithm pairs for each function. WOA+SA showed statistically significant improvements (p < 0.05) in the majority of comparisons, especially against the original WOA.

8. Extended Experiment with Heuristic Algorithms

Another experiment is conducted where WOA+SA is compared with seven well-known heuristic algorithms:

Genetic Algorithm (GA)

Grey Wolf Optimizer (GWO)

Dragonfly Algorithm (DA)

Chimp Optimization Algorithm (CHAO)

Fox Optimization Algorithm (FOA)

Particle Swarm Optimization (PSO)

Butterfly Optimization Algorithm (BOA)

All algorithms are tested using the same experimental configuration on benchmark suites CEC-2014, CEC-2017, CEC-2020, and CEC-2022. Wilcoxon signed-rank tests and statistical performance analyses (mean, standard deviation, and rank) are performed across all functions and datasets.

WOA+SA again consistently performed competitively, achieving top rankings and statistically significant advantages in many functions. This confirms the hybrid's capability beyond WOA variants and against a broad range of optimization strategies.

9. Results Summary

A summary of findings is presented below:

WOA+SA outperformed other WOA variants on a majority of functions.

Consistent convergence patterns demonstrate improved stability.

Statistically significant gains observed in both Wilcoxon and mean-rank analyses.

WOA+SA also proved competitive when compared to diverse heuristic algorithms like GWO, PSO, and BOA, often ranking among the top algorithms.

10. Conclusion

This study demonstrates that incorporating Simulated Annealing into the Whale Optimization Algorithm significantly improves its convergence behavior and robustness. WOA+SA consistently outperforms other WOA variants across multiple benchmark suites. The convergence plots clearly show more stable and deeper convergence trends.

Furthermore, the extended comparison with seven popular heuristics reveals WOA+SA as a strong and competitive optimization algorithm. It maintained top performance in statistical measures and Wilcoxon comparisons, showcasing its generalizability and efficiency.

These findings suggest that WOA+SA is not only effective in overcoming WOA’s weaknesses but also stands strong against a variety of modern metaheuristic techniques. This hybrid approach could be extended further for solving real-world optimization problems in engineering, logistics, and machine learning.

11. Future Work

Extend WOA+SA to constrained and dynamic optimization.

Explore hybridization with adaptive SA.

Apply to real-world engineering optimization problems.

Test scalability to higher dimensions (e.g., 100-D).

Study energy efficiency and runtime performance.
