from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import pulp

from src.utils import (
    is_task_individually_feasible,
    total_partition_cost,
)


@dataclass
class ILPResult:
    best_cost: float | None
    best_bins: list[list[int]]
    elapsed_seconds: float
    feasible: bool
    status: str


def solve_ilp(
    tasks: list[dict],
    servers: list[dict],
    max_bins: int | None = None,
) -> ILPResult:
    start = time.perf_counter()

    n = len(tasks)
    m = len(servers)

    if max_bins is None:
        max_bins = n  # Upper bound

    if not tasks:
        return ILPResult(
            best_cost=0.0,
            best_bins=[],
            elapsed_seconds=0.0,
            feasible=True,
            status="optimal",
        )

    # Feasibility check
    if any(not is_task_individually_feasible(task, servers) for task in tasks):
        elapsed = time.perf_counter() - start
        return ILPResult(
            best_cost=None,
            best_bins=[],
            elapsed_seconds=elapsed,
            feasible=False,
            status="infeasible",
        )

    # Create the problem
    prob = pulp.LpProblem("BinPacking", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("x", (range(n), range(max_bins)), cat="Binary")
    y = pulp.LpVariable.dicts("y", (range(max_bins), range(m)), cat="Binary")
    z = pulp.LpVariable.dicts("z", range(max_bins), cat="Binary")

    # Objective
    prob += pulp.lpSum(y[j][k] * servers[k]["cost"] for j in range(max_bins) for k in range(m))

    # Constraints
    # Each task assigned to exactly one bin
    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(max_bins)) == 1

    # Bin usage
    for j in range(max_bins):
        prob += pulp.lpSum(x[i][j] for i in range(n)) <= n * z[j]
        prob += pulp.lpSum(y[j][k] for k in range(m)) == z[j]

    # Capacity constraints
    bigM_cpu = sum(task["cpu"] for task in tasks)
    bigM_ram = sum(task["ram"] for task in tasks)

    for j in range(max_bins):
        for k in range(m):
            prob += pulp.lpSum(x[i][j] * tasks[i]["cpu"] for i in range(n)) <= y[j][k] * servers[k]["cpu"] + (1 - y[j][k]) * bigM_cpu
            prob += pulp.lpSum(x[i][j] * tasks[i]["ram"] for i in range(n)) <= y[j][k] * servers[k]["ram"] + (1 - y[j][k]) * bigM_ram

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    elapsed = time.perf_counter() - start

    if pulp.LpStatus[status] == "Optimal":
        # Extract solution
        best_bins = [[] for _ in range(max_bins)]
        for i in range(n):
            for j in range(max_bins):
                if pulp.value(x[i][j]) == 1:
                    best_bins[j].append(i)
                    break
        best_bins = [bin_indices for bin_indices in best_bins if bin_indices]  # Remove empty bins

        best_cost = total_partition_cost(tasks, servers, best_bins)
        return ILPResult(
            best_cost=best_cost,
            best_bins=best_bins,
            elapsed_seconds=elapsed,
            feasible=True,
            status="optimal",
        )
    else:
        return ILPResult(
            best_cost=None,
            best_bins=[],
            elapsed_seconds=elapsed,
            feasible=False,
            status=pulp.LpStatus[status],
        )