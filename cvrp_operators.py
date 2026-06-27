import copy
import random
from typing import Dict, List, Optional, Tuple

from cvrp_utils import (
    CVRPInstance,
    route_demand,
    is_solution_feasible,
)


Move = Tuple[str, Tuple]


def copy_solution(solution: List[List[int]]) -> List[List[int]]:
    """Deep copy a CVRP solution."""
    return [route[:] for route in solution]


def route_customer_positions(route: List[int]) -> List[int]:
    """
    Return positions of real customers inside a route.

    Example:
    route = [0, 5, 2, 0]
    returns [1, 2]
    """
    return [i for i, node in enumerate(route) if node != 0]


def non_empty_route_indices(solution: List[List[int]]) -> List[int]:
    """Return route indices that contain at least one customer."""
    return [
        i for i, route in enumerate(solution)
        if any(node != 0 for node in route)
    ]


def normalize_empty_route(route: List[int]) -> List[int]:
    """
    If a route has no customers, represent it as [0, 0].
    Otherwise, keep it as [0, ..., 0].
    """
    customers = [node for node in route if node != 0]

    if not customers:
        return [0, 0]

    return [0] + customers + [0]



def relocate_customer(
    solution: List[List[int]],
    instance: CVRPInstance,
    rng: random.Random,
    max_attempts: int = 100
) -> Optional[Tuple[List[List[int]], Move]]:
    """
    Move one customer from one route to another route.

    Returns:
        (new_solution, move_description)

    Move description example:
        ("relocate", (customer, source_route, target_route))
    """
    route_indices = non_empty_route_indices(solution)

    if not route_indices:
        return None

    for _ in range(max_attempts):
        new_solution = copy_solution(solution)

        source_route_index = rng.choice(route_indices)
        source_route = new_solution[source_route_index]

        customer_positions = route_customer_positions(source_route)

        if not customer_positions:
            continue

        customer_pos = rng.choice(customer_positions)
        customer = source_route[customer_pos]
        customer_demand = instance.demands[customer]

        possible_targets = list(range(len(new_solution)))
        rng.shuffle(possible_targets)

        for target_route_index in possible_targets:
            if target_route_index == source_route_index:
                continue

            target_route = new_solution[target_route_index]
            target_demand = route_demand(target_route, instance.demands)

            if target_demand + customer_demand > instance.capacity:
                continue

            # Remove customer from source.
            source_route.pop(customer_pos)
            new_solution[source_route_index] = normalize_empty_route(source_route)

            # Insert customer into target before final depot.
            target_route = new_solution[target_route_index]
            insert_pos = rng.randint(1, len(target_route) - 1)
            target_route.insert(insert_pos, customer)
            new_solution[target_route_index] = normalize_empty_route(target_route)

            feasible, _ = is_solution_feasible(new_solution, instance)

            if feasible:
                move = (
                    "relocate",
                    (customer, source_route_index, target_route_index)
                )
                return new_solution, move

    return None


def swap_customers(
    solution: List[List[int]],
    instance: CVRPInstance,
    rng: random.Random,
    max_attempts: int = 100
) -> Optional[Tuple[List[List[int]], Move]]:
    """
    Swap two customers between two routes.

    Returns:
        (new_solution, move_description)

    Move description example:
        ("swap", (customer_a, customer_b, route_a, route_b))
    """
    route_indices = non_empty_route_indices(solution)

    if len(route_indices) < 2:
        return None

    for _ in range(max_attempts):
        new_solution = copy_solution(solution)

        route_a_index, route_b_index = rng.sample(route_indices, 2)

        route_a = new_solution[route_a_index]
        route_b = new_solution[route_b_index]

        positions_a = route_customer_positions(route_a)
        positions_b = route_customer_positions(route_b)

        if not positions_a or not positions_b:
            continue

        pos_a = rng.choice(positions_a)
        pos_b = rng.choice(positions_b)

        customer_a = route_a[pos_a]
        customer_b = route_b[pos_b]

        demand_a = instance.demands[customer_a]
        demand_b = instance.demands[customer_b]

        route_a_demand = route_demand(route_a, instance.demands)
        route_b_demand = route_demand(route_b, instance.demands)

        new_route_a_demand = route_a_demand - demand_a + demand_b
        new_route_b_demand = route_b_demand - demand_b + demand_a

        if new_route_a_demand > instance.capacity:
            continue

        if new_route_b_demand > instance.capacity:
            continue

        route_a[pos_a] = customer_b
        route_b[pos_b] = customer_a

        feasible, _ = is_solution_feasible(new_solution, instance)

        if feasible:
            move = (
                "swap",
                (customer_a, customer_b, route_a_index, route_b_index)
            )
            return new_solution, move

    return None


def two_opt_route(
    solution: List[List[int]],
    instance: CVRPInstance,
    rng: random.Random,
    max_attempts: int = 100
) -> Optional[Tuple[List[List[int]], Move]]:
    """
    Apply 2-opt inside one route by reversing a customer segment.

    This never changes capacity or customer assignment.
    It only changes order inside a route.

    Returns:
        (new_solution, move_description)

    Move description example:
        ("two_opt", (route_index, i, j))
    """
    candidate_routes = [
        i for i, route in enumerate(solution)
        if len(route_customer_positions(route)) >= 2
    ]

    if not candidate_routes:
        return None

    for _ in range(max_attempts):
        new_solution = copy_solution(solution)

        route_index = rng.choice(candidate_routes)
        route = new_solution[route_index]

        # Customer positions are from 1 to len(route)-2.
        customer_positions = route_customer_positions(route)

        if len(customer_positions) < 2:
            continue

        i, j = sorted(rng.sample(customer_positions, 2))

        if i == j:
            continue

        route[i:j + 1] = reversed(route[i:j + 1])

        feasible, _ = is_solution_feasible(new_solution, instance)

        if feasible:
            move = ("two_opt", (route_index, i, j))
            return new_solution, move

    return None


def random_neighbor(
    solution: List[List[int]],
    instance: CVRPInstance,
    rng: random.Random,
    operator_weights: Optional[Dict[str, float]] = None
) -> Optional[Tuple[List[List[int]], Move]]:
    """
    Generate one random feasible neighbor.

    This function chooses between:
    - relocate
    - swap
    - two_opt

    It will be used later by Simulated Annealing and Tabu Search.
    """
    if operator_weights is None:
        operator_weights = {
            "relocate": 1.0,
            "swap": 1.0,
            "two_opt": 1.0,
        }

    operators = list(operator_weights.keys())
    weights = list(operator_weights.values())

    for _ in range(20):
        chosen_operator = rng.choices(operators, weights=weights, k=1)[0]

        if chosen_operator == "relocate":
            result = relocate_customer(solution, instance, rng)

        elif chosen_operator == "swap":
            result = swap_customers(solution, instance, rng)

        elif chosen_operator == "two_opt":
            result = two_opt_route(solution, instance, rng)

        else:
            raise ValueError(f"Unknown operator: {chosen_operator}")

        if result is not None:
            return result

    return None