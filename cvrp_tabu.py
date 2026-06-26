import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from cvrp_utils import (
    CVRPInstance,
    build_distance_matrix,
    greedy_initial_solution,
    is_solution_feasible,
    total_cost,
)

from cvrp_operators import random_neighbor, Move


@dataclass
class TabuResult:
    best_solution: List[List[int]]
    best_cost: float
    initial_cost: float
    elapsed_time: float
    iterations: int
    improved_moves: int
    tabu_skipped_moves: int
    history: List[float]


def tabu_key(move: Move) -> Tuple:
    """
    Convert a move into a simple tabu key.

    We do not want to store the exact route positions only,
    because after a few moves the same customer can appear in different places.

    So:
    - relocate: make the moved customer tabu for relocation
    - swap: make the swapped customer pair tabu
    - two_opt: make this route-level reversal tabu
    """
    move_type, data = move

    if move_type == "relocate":
        customer, source_route, target_route = data
        return ("relocate", customer)

    if move_type == "swap":
        customer_a, customer_b, route_a, route_b = data
        return ("swap", tuple(sorted((customer_a, customer_b))))

    if move_type == "two_opt":
        route_index, i, j = data
        return ("two_opt", route_index, i, j)

    return move


def tabu_search(
    instance: CVRPInstance,
    initial_solution: Optional[List[List[int]]] = None,
    max_iterations: int = 1000,
    neighborhood_sample_size: int = 80,
    tabu_tenure: int = 20,
    seed: int = 42,
    operator_weights: Optional[Dict[str, float]] = None,
) -> TabuResult:
    """
    Tabu Search for CVRP.

    Logic:
    - Start from a feasible solution.
    - At each iteration, sample many neighbors.
    - Choose the best non-tabu neighbor.
    - If a tabu neighbor gives a new global best, allow it anyway
      using aspiration criterion.
    - Store recent move keys in tabu memory.
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

    if operator_weights is None:
        operator_weights = {
            "relocate": 1.0,
            "swap": 1.0,
            "two_opt": 2.0,
        }

    tabu_queue = deque()
    tabu_set = set()

    improved_moves = 0
    tabu_skipped_moves = 0
    history = [best_cost]

    for iteration in range(1, max_iterations + 1):
        best_candidate_solution = None
        best_candidate_cost = float("inf")
        best_candidate_move = None
        best_candidate_key = None

        for _ in range(neighborhood_sample_size):
            result = random_neighbor(
                solution=current_solution,
                instance=instance,
                rng=rng,
                operator_weights=operator_weights,
            )

            if result is None:
                continue

            neighbor_solution, move = result
            neighbor_cost = total_cost(neighbor_solution, dist)
            key = tabu_key(move)

            is_tabu = key in tabu_set

            # Aspiration criterion:
            # even if the move is tabu, allow it if it beats the global best.
            allowed_by_aspiration = neighbor_cost < best_cost

            if is_tabu and not allowed_by_aspiration:
                tabu_skipped_moves += 1
                continue

            if neighbor_cost < best_candidate_cost:
                best_candidate_solution = neighbor_solution
                best_candidate_cost = neighbor_cost
                best_candidate_move = move
                best_candidate_key = key

        # If all sampled moves were tabu or invalid, just continue.
        if best_candidate_solution is None:
            history.append(best_cost)
            continue

        # Move to best allowed candidate, even if it is worse than current.
        current_solution = best_candidate_solution
        current_cost = best_candidate_cost

        # Add move to tabu memory.
        tabu_queue.append(best_candidate_key)
        tabu_set.add(best_candidate_key)

        if len(tabu_queue) > tabu_tenure:
            old_key = tabu_queue.popleft()
            tabu_set.discard(old_key)

        if current_cost < best_cost:
            best_solution = [route[:] for route in current_solution]
            best_cost = current_cost
            improved_moves += 1

        history.append(best_cost)

    elapsed_time = time.time() - start_time

    return TabuResult(
        best_solution=best_solution,
        best_cost=best_cost,
        initial_cost=initial_cost,
        elapsed_time=elapsed_time,
        iterations=max_iterations,
        improved_moves=improved_moves,
        tabu_skipped_moves=tabu_skipped_moves,
        history=history,
    )