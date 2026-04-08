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

task_sizes = [3,5,10]

ilp_costs, sa_costs = [], []
ilp_times, sa_times = [], []
ilp_statuses = []

for size in task_sizes:
    print(f"\nRunning for {size} tasks...")

    tasks, servers = prepare_data(df, num_tasks=size)
    
    sa_result = solve_sa(tasks, servers)
    sa_costs.append(sa_result.best_cost)
    sa_times.append(sa_result.elapsed_seconds)

    ilp_result = solve_ilp(tasks, servers)
    ilp_costs.append(ilp_result.best_cost)
    ilp_times.append(ilp_result.elapsed_seconds)
    ilp_statuses.append(ilp_result.status)

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
print("ilp_costs:", ilp_costs)
print("ilp_times:", ilp_times)
print("ilp_statuses:", ilp_statuses)

# Terminal comparison table
print("\nComparison table:")
header = (
    f"{'Tasks':<8}{'SA Cost':<15}{'ILP Cost':<15}{'Gap (SA-ILP)':<18}"
    f"{'SA Time (s)':<15}{'ILP Time (s)':<15}{'ILP status':<14}"
)
print(header)
print("-" * len(header))
for size, sa_cost, ilp_cost, sa_time, ilp_time, ilp_status in zip(
    task_sizes, sa_costs, ilp_costs, sa_times, ilp_times, ilp_statuses
):
    sa_cost_str = f"{sa_cost:.6f}" if sa_cost is not None else "N/A"
    ilp_cost_str = f"{ilp_cost:.6f}" if ilp_cost is not None else "N/A"
    sa_time_str = f"{sa_time:.6f}" if sa_time is not None else "N/A"
    ilp_time_str = f"{ilp_time:.6f}" if ilp_time is not None else "N/A"
    if sa_cost is not None and ilp_cost is not None:
        gap_str = f"{sa_cost - ilp_cost:.6f}"
    else:
        gap_str = "N/A"
    print(
        f"{size:<8}{sa_cost_str:<15}{ilp_cost_str:<15}{gap_str:<18}"
        f"{sa_time_str:<15}{ilp_time_str:<15}{ilp_status:<14}"
    )

# Save plots
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

sa_cost_points = [(size, cost) for size, cost in zip(task_sizes, sa_costs) if cost is not None]
ilp_cost_points = [(size, cost) for size, cost in zip(task_sizes, ilp_costs) if cost is not None]
sa_time_points = [(size, run_time) for size, run_time in zip(task_sizes, sa_times) if run_time is not None]
ilp_time_points = [(size, run_time) for size, run_time in zip(task_sizes, ilp_times) if run_time is not None]

# Cost comparison graph
plt.figure(figsize=(8, 5))
if sa_cost_points:
    sa_x = [x - 0.03 for x, _ in sa_cost_points]
    sa_y = [y for _, y in sa_cost_points]
    plt.plot(
        sa_x,
        sa_y,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
        linewidth=2,
        linestyle="--",
        label="SA",
        zorder=3,
    )
if ilp_cost_points:
    ilp_x = [x + 0.03 for x, _ in ilp_cost_points]
    ilp_y = [y for _, y in ilp_cost_points]
    plt.plot(
        ilp_x,
        ilp_y,
        marker="s",
        markersize=7,
        linewidth=2,
        alpha=0.9,
        label="ILP",
        zorder=2,
    )
plt.title("Cost comparison: SA vs ILP")
plt.xlabel("Number of tasks")
plt.ylabel("Total cost (normalized)")
plt.xticks(task_sizes)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(figures_dir / "cost_comparison.png", dpi=200)
plt.close()

# Runtime comparison graph
plt.figure(figsize=(8, 5))
if sa_time_points:
    sa_t_x = [x - 0.03 for x, _ in sa_time_points]
    sa_t_y = [y for _, y in sa_time_points]
    plt.plot(
        sa_t_x,
        sa_t_y,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
        linewidth=2,
        linestyle="--",
        label="SA",
        zorder=3,
    )
if ilp_time_points:
    ilp_t_x = [x + 0.03 for x, _ in ilp_time_points]
    ilp_t_y = [y for _, y in ilp_time_points]
    plt.plot(
        ilp_t_x,
        ilp_t_y,
        marker="s",
        markersize=7,
        linewidth=2,
        alpha=0.9,
        label="ILP",
        zorder=2,
    )
plt.title("Runtime comparison: SA vs ILP")
plt.xlabel("Number of tasks")
plt.ylabel("Runtime (seconds)")
plt.xticks(task_sizes)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(figures_dir / "runtime_comparison.png", dpi=200)
plt.close()

print(f"Saved plots to: {figures_dir.resolve()}")