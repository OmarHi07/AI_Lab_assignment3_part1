import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from cvrp_utils import (
    CVRPInstance,
    build_distance_matrix,
    greedy_initial_solution,
    is_solution_feasible,
    total_cost,
)

from cvrp_operators import random_neighbor


@dataclass
class SAResult:
    best_solution: List[List[int]]
    best_cost: float
    initial_cost: float
    elapsed_time: float
    iterations: int
    accepted_moves: int
    improved_moves: int
    history: List[float]


def simulated_annealing(
    instance: CVRPInstance,
    initial_solution: Optional[List[List[int]]] = None,
    max_iterations: int = 10000,
    initial_temperature: Optional[float] = None,
    final_temperature: float = 1e-3,
    cooling_rate: float = 0.995,
    seed: int = 42,
    operator_weights: Optional[Dict[str, float]] = None,
) -> SAResult:
    """
    Simulated Annealing for CVRP.

    Logic:
    - Start from a feasible solution.
    - Generate a random feasible neighbor.
    - If neighbor is better, accept it.
    - If neighbor is worse, accept it with probability exp(-delta / T).
    - Gradually reduce temperature.

    This is our first real metaheuristic optimizer.
    """
    start_time = time.time()
    rng = random.Random(seed)

    dist = build_distance_matrix(instance)

    if initial_solution is None:
        current_solution = greedy_initial_solution(instance)
    else:
        current_solution = [route[:] for route in initial_solution]

    feasible, errors = is_solution_feasible(current_solution, instance)

    if not feasible:
        raise ValueError(f"Initial solution is infeasible: {errors}")

    current_cost = total_cost(current_solution, dist)

    best_solution = [route[:] for route in current_solution]
    best_cost = current_cost
    initial_cost = current_cost

    if initial_temperature is None:
        # A reasonable automatic starting temperature.
        # Bigger instances/costs get bigger initial temperature.
        temperature = max(1.0, 0.05 * current_cost)
    else:
        temperature = initial_temperature

    if operator_weights is None:
        operator_weights = {
            "relocate": 1.0,
            "swap": 1.0,
            "two_opt": 2.0,
        }

    accepted_moves = 0
    improved_moves = 0
    history = [best_cost]

    iteration = 0

    while iteration < max_iterations and temperature > final_temperature:
        iteration += 1

        result = random_neighbor(
            solution=current_solution,
            instance=instance,
            rng=rng,
            operator_weights=operator_weights,
        )

        if result is None:
            temperature *= cooling_rate
            history.append(best_cost)
            continue

        neighbor_solution, move = result
        neighbor_cost = total_cost(neighbor_solution, dist)

        delta = neighbor_cost - current_cost

        accept = False

        if delta < 0:
            # Better solution: always accept.
            accept = True
            improved_moves += 1
        else:
            # Worse solution: accept with Boltzmann probability.
            probability = math.exp(-delta / temperature)

            if rng.random() < probability:
                accept = True

        if accept:
            current_solution = neighbor_solution
            current_cost = neighbor_cost
            accepted_moves += 1

            if current_cost < best_cost:
                best_solution = [route[:] for route in current_solution]
                best_cost = current_cost

        temperature *= cooling_rate
        history.append(best_cost)

    elapsed_time = time.time() - start_time

    return SAResult(
        best_solution=best_solution,
        best_cost=best_cost,
        initial_cost=initial_cost,
        elapsed_time=elapsed_time,
        iterations=iteration,
        accepted_moves=accepted_moves,
        improved_moves=improved_moves,
        history=history,
    )