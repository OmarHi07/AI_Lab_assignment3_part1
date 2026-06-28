from typing import List, Optional, Tuple

from cvrp_utils import (
    CVRPInstance,
    route_demand,
    route_cost,
    total_cost,
    improve_solution_routes_2opt,
    order_route_nearest_neighbor,
    two_opt_improve_route,
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


def build_improved_route_from_customers(
    customers: List[int],
    dist: List[List[float]],
) -> List[int]:
    """
    Build a route from a customer group and improve it with 2-opt.
    """
    if not customers:
        return [0, 0]

    route = order_route_nearest_neighbor(customers, dist)
    route = two_opt_improve_route(route, dist)

    return route


def best_two_route_repair_once(
    solution: List[List[int]],
    instance: CVRPInstance,
    dist: List[List[float]],
    max_combined_customers: int = 14,
) -> Optional[List[List[int]]]:
    """
    Best-improvement two-route repair.

    Pick two routes, combine their customers, and try many feasible ways
    to split them back into two routes.

    This is stronger than simple relocate/swap because it can move several
    customers between two routes at once.
    """
    best_delta = 0.0
    best_solution = None

    for route_a_idx in range(len(solution)):
        route_a = solution[route_a_idx]
        customers_a = [node for node in route_a if node != 0]

        if not customers_a:
            continue

        for route_b_idx in range(route_a_idx + 1, len(solution)):
            route_b = solution[route_b_idx]
            customers_b = [node for node in route_b if node != 0]

            if not customers_b:
                continue

            combined_customers = customers_a + customers_b
            combined_count = len(combined_customers)

            # Avoid exponential explosion on very large combined route pairs.
            if combined_count > max_combined_customers:
                continue

            old_cost = route_cost(route_a, dist) + route_cost(route_b, dist)

            first_customer = combined_customers[0]
            remaining_customers = combined_customers[1:]

            number_of_remaining = len(remaining_customers)

            # Force first_customer to be in group A to avoid duplicate symmetric splits.
            for mask in range(1 << number_of_remaining):
                group_a = [first_customer]
                group_b = []

                for bit_index, customer in enumerate(remaining_customers):
                    if mask & (1 << bit_index):
                        group_a.append(customer)
                    else:
                        group_b.append(customer)

                demand_a = sum(instance.demands[c] for c in group_a)
                demand_b = sum(instance.demands[c] for c in group_b)

                if demand_a > instance.capacity:
                    continue

                if demand_b > instance.capacity:
                    continue

                new_route_a = build_improved_route_from_customers(group_a, dist)
                new_route_b = build_improved_route_from_customers(group_b, dist)

                new_cost = route_cost(new_route_a, dist) + route_cost(new_route_b, dist)

                delta = new_cost - old_cost

                if delta < best_delta - 1e-9:
                    candidate_solution = copy_solution(solution)
                    candidate_solution[route_a_idx] = new_route_a
                    candidate_solution[route_b_idx] = new_route_b

                    best_delta = delta
                    best_solution = candidate_solution

    return best_solution


def two_route_repair_local_search(
    solution: List[List[int]],
    instance: CVRPInstance,
    dist: List[List[float]],
    max_passes: int = 10,
    max_combined_customers: int = 14,
) -> List[List[int]]:
    """
    Repeatedly apply two-route repair until no improving pair is found.

    This is useful after ALNS, especially on X-n101-k25, because it can fix
    poor route grouping that simple relocate/swap may not fix.
    """
    current = copy_solution(solution)
    current = improve_solution_routes_2opt(current, dist)

    current_cost = total_cost(current, dist)

    for _ in range(max_passes):
        candidate = best_two_route_repair_once(
            current,
            instance,
            dist,
            max_combined_customers=max_combined_customers,
        )

        if candidate is None:
            break

        candidate = improve_solution_routes_2opt(candidate, dist)
        candidate_cost = total_cost(candidate, dist)

        if candidate_cost < current_cost - 1e-9:
            current = candidate
            current_cost = candidate_cost
        else:
            break

    return current