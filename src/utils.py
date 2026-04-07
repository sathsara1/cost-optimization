from __future__ import annotations

from typing import Iterable


def aggregate_demands(tasks: list[dict], bin_indices: Iterable[int]) -> tuple[int, int]:
    cpu = 0
    ram = 0
    for idx in bin_indices:
        cpu += int(tasks[idx]["cpu"])
        ram += int(tasks[idx]["ram"])
    return cpu, ram


def cheapest_fitting_cost(tasks: list[dict], servers: list[dict], bin_indices: Iterable[int]) -> float | None:
    cpu_demand, ram_demand = aggregate_demands(tasks, bin_indices)
    feasible_costs = [
        float(server["cost"])
        for server in servers
        if int(server["cpu"]) >= cpu_demand and int(server["ram"]) >= ram_demand
    ]
    if not feasible_costs:
        return None
    return min(feasible_costs)


def total_partition_cost(tasks: list[dict], servers: list[dict], bins: list[list[int]]) -> float | None:
    total = 0.0
    for bin_indices in bins:
        if not bin_indices:
            continue
        bin_cost = cheapest_fitting_cost(tasks, servers, bin_indices)
        if bin_cost is None:
            return None
        total += bin_cost
    return total


def is_task_individually_feasible(task: dict, servers: list[dict]) -> bool:
    task_cpu = int(task["cpu"])
    task_ram = int(task["ram"])
    return any(
        int(server["cpu"]) >= task_cpu and int(server["ram"]) >= task_ram
        for server in servers
    )


def singleton_partition(task_count: int) -> list[list[int]]:
    return [[idx] for idx in range(task_count)]
