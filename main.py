import matplotlib.pyplot as plt
import mlcroissant as mlc
import pandas as pd
from pathlib import Path

from src.preprocess import prepare_data
from src.sa_solver import solve_sa

# Load dataset via Croissant
croissant_dataset = mlc.Dataset(
    "https://www.kaggle.com/datasets/freshersstaff/multi-cloud-resource-dataset/croissant/download"
)
record_sets = croissant_dataset.metadata.record_sets
df = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))
print('Columns:', df.columns.tolist())

task_sizes = [10, 30, 50]

ilp_costs, sa_costs = [], []
ilp_times, sa_times = [], []

for size in task_sizes:
    print(f"\nRunning for {size} tasks...")

    tasks, servers = prepare_data(df, num_tasks=size)
    sa_result = solve_sa(tasks, servers)
    sa_costs.append(sa_result.best_cost)
    sa_times.append(sa_result.elapsed_seconds)

    # Placeholder until ILP solver is added.
    ilp_costs.append(None)
    ilp_times.append(None)

    print(
        f"SA | feasible={sa_result.feasible} | "
        f"best_cost={sa_result.best_cost} | "
        f"time={sa_result.elapsed_seconds:.4f}s | "
        f"iterations={sa_result.iterations}"
    )

print("\nFinal comparison arrays:")
print("task_sizes:", task_sizes)
print("sa_costs:", sa_costs)
print("sa_times:", sa_times)

