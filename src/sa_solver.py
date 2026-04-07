from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from src.utils import (
    is_task_individually_feasible,
    singleton_partition,
    total_partition_cost,
)


@dataclass
class SAResult:
    best_cost: float | None
    best_bins: list[list[int]]
    elapsed_seconds: float
    iterations: int
    accepted_moves: int
    feasible: bool
    seed: int | None
    stopped_early: bool
    no_improve_iterations: int


def _remove_empty_bins(bins: list[list[int]]) -> list[list[int]]:
    return [bin_indices for bin_indices in bins if bin_indices]


def _generate_neighbor(bins: list[list[int]], rng: random.Random) -> list[list[int]]:
    candidate = [bin_indices[:] for bin_indices in bins]
    if not candidate:
        return candidate

    from_bin_idx = rng.randrange(len(candidate))
    while not candidate[from_bin_idx]:
        from_bin_idx = rng.randrange(len(candidate))

    pos = rng.randrange(len(candidate[from_bin_idx]))
    task_idx = candidate[from_bin_idx].pop(pos)
    candidate = _remove_empty_bins(candidate)

    # Either place task in existing bin or create new one.
    create_new_bin = not candidate or rng.random() < 0.25
    if create_new_bin:
        candidate.append([task_idx])
    else:
        to_bin_idx = rng.randrange(len(candidate))
        candidate[to_bin_idx].append(task_idx)

    return _remove_empty_bins(candidate)


def solve_sa(
    tasks: list[dict],
    servers: list[dict],
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.95,
    min_temperature: float = 0.01,
    iterations_per_temp: int = 200,
    random_seed: int | None = 42,
    early_stop_no_improve_iters: int | None = 150,
    min_iterations_before_stop: int = 300,
) -> SAResult:
    start = time.perf_counter()
    rng = random.Random(random_seed)

    if not tasks:
        return SAResult(
            best_cost=0.0,
            best_bins=[],
            elapsed_seconds=0.0,
            iterations=0,
            accepted_moves=0,
            feasible=True,
            seed=random_seed,
            stopped_early=False,
            no_improve_iterations=0,
        )

    #  feasibility guard (each task must fit at least one server type)
    if any(not is_task_individually_feasible(task, servers) for task in tasks):
        elapsed = time.perf_counter() - start
        return SAResult(
            best_cost=None,
            best_bins=[],
            elapsed_seconds=elapsed,
            iterations=0,
            accepted_moves=0,
            feasible=False,
            seed=random_seed,
            stopped_early=False,
            no_improve_iterations=0,
        )

    current_bins = singleton_partition(len(tasks))
    current_cost = total_partition_cost(tasks, servers, current_bins)
    if current_cost is None:
        elapsed = time.perf_counter() - start
        return SAResult(
            best_cost=None,
            best_bins=[],
            elapsed_seconds=elapsed,
            iterations=0,
            accepted_moves=0,
            feasible=False,
            seed=random_seed,
            stopped_early=False,
            no_improve_iterations=0,
        )

    best_bins = [bin_indices[:] for bin_indices in current_bins]
    best_cost = current_cost
    temperature = initial_temperature
    iterations = 0
    accepted_moves = 0
    no_improve_counter = 0
    stopped_early = False

    while temperature > min_temperature:
        for _ in range(iterations_per_temp):
            iterations += 1
            improved_this_iteration = False
            neighbor_bins = _generate_neighbor(current_bins, rng)
            neighbor_cost = total_partition_cost(tasks, servers, neighbor_bins)

            # Feasible only (if not feasible, skip)
            if neighbor_cost is None:
                continue

            delta = neighbor_cost - current_cost
            if delta <= 0:
                accept = True
            else:
                accept_probability = math.exp(-delta / max(temperature, 1e-9))
                accept = rng.random() < accept_probability

            if accept:
                current_bins = neighbor_bins
                current_cost = neighbor_cost
                accepted_moves += 1

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_bins = [bin_indices[:] for bin_indices in current_bins]
                    improved_this_iteration = True

            if improved_this_iteration:
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if (
                early_stop_no_improve_iters is not None
                and iterations >= min_iterations_before_stop
                and no_improve_counter >= early_stop_no_improve_iters
            ):
                stopped_early = True
                break

        if stopped_early:
            break
        temperature *= cooling_rate

    elapsed = time.perf_counter() - start
    return SAResult(
        best_cost=best_cost,
        best_bins=best_bins,
        elapsed_seconds=elapsed,
        iterations=iterations,
        accepted_moves=accepted_moves,
        feasible=True,
        seed=random_seed,
        stopped_early=stopped_early,
        no_improve_iterations=no_improve_counter,
    )
