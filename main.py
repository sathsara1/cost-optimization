import matplotlib.pyplot as plt
import mlcroissant as mlc
import pandas as pd
from pathlib import Path

from src.preprocess import prepare_data
from src.sa_solver import solve_sa
from src.ilp_solver import solve_ilp

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

    ilp_result = solve_ilp(tasks, servers)
    ilp_costs.append(ilp_result.best_cost)
    ilp_times.append(ilp_result.elapsed_seconds)

    print(
        f"SA | feasible={sa_result.feasible} | "
        f"best_cost={sa_result.best_cost} | "
        f"time={sa_result.elapsed_seconds:.4f}s | "
        f"iterations={sa_result.iterations}"
    )
    print(
        f"ILP | feasible={ilp_result.feasible} | "
        f"best_cost={ilp_result.best_cost} | "
        f"time={ilp_result.elapsed_seconds:.4f}s | "
        f"status={ilp_result.status}"
    )

print("\nFinal comparison arrays:")
print("task_sizes:", task_sizes)
print("sa_costs:", sa_costs)
print("sa_times:", sa_times)

