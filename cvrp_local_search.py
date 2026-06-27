from typing import List, Optional, Tuple

from cvrp_utils import (
    CVRPInstance,
    route_demand,
    route_cost,
    total_cost,
    improve_solution_routes_2opt,
)

from cvrp_operators import copy_solution, normalize_empty_route


def best_relocate_once(
    solution: List[List[int]],
    instance: CVRPInstance,
    dist: List[List[float]],
) -> Optional[List[List[int]]]:
    """
    Best-improvement cross-route relocate.

    Move one customer from one route to another route if it improves cost.
    """
    best_delta = 0.0
    best_move = None

    route_demands = [
        route_demand(route, instance.demands)
        for route in solution
    ]

    for source_idx, source_route in enumerate(solution):
        if len(source_route) <= 3:
            continue

        for pos in range(1, len(source_route) - 1):
            customer = source_route[pos]
            customer_demand = instance.demands[customer]

            prev_node = source_route[pos - 1]
            next_node = source_route[pos + 1]

            remove_delta = (
                dist[prev_node][next_node]
                - dist[prev_node][customer]
                - dist[customer][next_node]
            )

            for target_idx, target_route in enumerate(solution):
                if target_idx == source_idx:
                    continue

                if route_demands[target_idx] + customer_demand > instance.capacity:
                    continue

                for insert_pos in range(1, len(target_route)):
                    before = target_route[insert_pos - 1]
                    after = target_route[insert_pos]

                    insert_delta = (
                        dist[before][customer]
                        + dist[customer][after]
                        - dist[before][after]
                    )

                    delta = remove_delta + insert_delta

                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_move = (source_idx, pos, target_idx, insert_pos, customer)

    if best_move is None:
        return None

    source_idx, pos, target_idx, insert_pos, customer = best_move

    new_solution = copy_solution(solution)

    new_solution[source_idx].pop(pos)
    new_solution[source_idx] = normalize_empty_route(new_solution[source_idx])

    # If source route was before target route and changed length, target route object is still okay
    new_solution[target_idx].insert(insert_pos, customer)
    new_solution[target_idx] = normalize_empty_route(new_solution[target_idx])

    return new_solution


def best_swap_once(
    solution: List[List[int]],
    instance: CVRPInstance,
    dist: List[List[float]],
) -> Optional[List[List[int]]]:
    """
    Best-improvement cross-route swap.
    """
    best_delta = 0.0
    best_move = None

    route_demands = [
        route_demand(route, instance.demands)
        for route in solution
    ]

    current_route_costs = [
        route_cost(route, dist)
        for route in solution
    ]

    for route_a_idx in range(len(solution)):
        route_a = solution[route_a_idx]

        for route_b_idx in range(route_a_idx + 1, len(solution)):
            route_b = solution[route_b_idx]

            if len(route_a) <= 2 or len(route_b) <= 2:
                continue

            for pos_a in range(1, len(route_a) - 1):
                customer_a = route_a[pos_a]
                demand_a = instance.demands[customer_a]

                for pos_b in range(1, len(route_b) - 1):
                    customer_b = route_b[pos_b]
                    demand_b = instance.demands[customer_b]

                    new_demand_a = route_demands[route_a_idx] - demand_a + demand_b
                    new_demand_b = route_demands[route_b_idx] - demand_b + demand_a

                    if new_demand_a > instance.capacity:
                        continue

                    if new_demand_b > instance.capacity:
                        continue

                    new_route_a = route_a[:]
                    new_route_b = route_b[:]

                    new_route_a[pos_a] = customer_b
                    new_route_b[pos_b] = customer_a

                    old_cost = (
                        current_route_costs[route_a_idx]
                        + current_route_costs[route_b_idx]
                    )

                    new_cost = (
                        route_cost(new_route_a, dist)
                        + route_cost(new_route_b, dist)
                    )

                    delta = new_cost - old_cost

                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_move = (
                            route_a_idx,
                            pos_a,
                            route_b_idx,
                            pos_b,
                        )

    if best_move is None:
        return None

    route_a_idx, pos_a, route_b_idx, pos_b = best_move

    new_solution = copy_solution(solution)

    new_solution[route_a_idx][pos_a], new_solution[route_b_idx][pos_b] = (
        new_solution[route_b_idx][pos_b],
        new_solution[route_a_idx][pos_a],
    )

    return new_solution


def local_search_improvement(
    solution: List[List[int]],
    instance: CVRPInstance,
    dist: List[List[float]],
    max_passes: int = 30,
) -> List[List[int]]:
    """
    Deterministic local search polish.

    Repeatedly applies:
    1. route-level 2-opt
    2. best cross-route relocate
    3. best cross-route swap
    """
    current = copy_solution(solution)
    current = improve_solution_routes_2opt(current, dist)

    current_cost = total_cost(current, dist)

    for _ in range(max_passes):
        improved = False

        for operator in [best_relocate_once, best_swap_once]:
            candidate = operator(current, instance, dist)

            if candidate is None:
                continue

            candidate = improve_solution_routes_2opt(candidate, dist)
            candidate_cost = total_cost(candidate, dist)

            if candidate_cost < current_cost - 1e-9:
                current = candidate
                current_cost = candidate_cost
                improved = True
                break

        if not improved:
            break

    return current