import math
from typing import List, Optional, Tuple

from cvrp_utils import (
    CVRPInstance,
    build_distance_matrix,
    route_cost,
    total_cost,
    is_solution_feasible,
    order_route_nearest_neighbor,
    two_opt_improve_route,
)


def build_route_from_customers(
    customers: List[int],
    dist: List[List[float]],
) -> List[int]:
    if not customers:
        return [0, 0]

    route = order_route_nearest_neighbor(customers, dist)
    route = two_opt_improve_route(route, dist)
    return route


def angle_order(
    instance: CVRPInstance,
    offset: float = 0.0,
    reverse: bool = False,
) -> List[int]:
    depot_x, depot_y = instance.coordinates[0]

    customers = [node for node in instance.coordinates.keys() if node != 0]

    def shifted_angle(customer: int) -> float:
        x, y = instance.coordinates[customer]
        angle = math.atan2(y - depot_y, x - depot_x)
        return (angle - offset) % (2 * math.pi)

    return sorted(customers, key=shifted_angle, reverse=reverse)


def nearest_neighbor_customer_order(
    instance: CVRPInstance,
    start_customer: int,
    dist: List[List[float]],
) -> List[int]:
    unvisited = set(instance.coordinates.keys()) - {0}
    order = []

    current = start_customer
    unvisited.remove(current)
    order.append(current)

    while unvisited:
        next_customer = min(
            unvisited,
            key=lambda customer: dist[current][customer],
        )

        unvisited.remove(next_customer)
        order.append(next_customer)
        current = next_customer

    return order


def split_order_exact_k(
    order: List[int],
    instance: CVRPInstance,
    dist: List[List[float]],
) -> Optional[List[List[int]]]:
    """
    Split one giant customer order into exactly K feasible routes.

    DP state:
        dp[i][r] = best cost to serve first i customers using r routes

    Transition:
        previous split j -> i, if demand(order[j:i]) <= capacity
    """
    n = len(order)
    k = instance.vehicle_count
    capacity = instance.capacity

    segment_cost = {}
    segment_route = {}

    for start in range(n):
        demand_sum = 0

        for end in range(start + 1, n + 1):
            customer = order[end - 1]
            demand_sum += instance.demands[customer]

            if demand_sum > capacity:
                break

            customers = order[start:end]
            route = build_route_from_customers(customers, dist)

            segment_cost[(start, end)] = route_cost(route, dist)
            segment_route[(start, end)] = route

    infinity = float("inf")

    dp = [
        [infinity] * (k + 1)
        for _ in range(n + 1)
    ]

    parent = [
        [None] * (k + 1)
        for _ in range(n + 1)
    ]

    dp[0][0] = 0.0

    for used_customers in range(n):
        for used_routes in range(k):
            if dp[used_customers][used_routes] == infinity:
                continue

            for next_index in range(used_customers + 1, n + 1):
                key = (used_customers, next_index)

                if key not in segment_cost:
                    break

                new_cost = (
                    dp[used_customers][used_routes]
                    + segment_cost[key]
                )

                if new_cost < dp[next_index][used_routes + 1]:
                    dp[next_index][used_routes + 1] = new_cost
                    parent[next_index][used_routes + 1] = used_customers

    if dp[n][k] == infinity:
        return None

    routes = []
    current_index = n
    current_routes = k

    while current_routes > 0:
        previous_index = parent[current_index][current_routes]

        if previous_index is None:
            return None

        routes.append(segment_route[(previous_index, current_index)])

        current_index = previous_index
        current_routes -= 1

    routes.reverse()

    feasible, errors = is_solution_feasible(routes, instance)

    if not feasible:
        return None

    return routes


def giant_split_candidates(
    instance: CVRPInstance,
) -> List[Tuple[str, List[List[int]], float]]:
    """
    Generate route-first / split-second initial solutions.

    This is especially useful for tight-capacity instances like X-n101-k25.
    """
    dist = build_distance_matrix(instance)

    orders = []

    # Sweep/angle orders with many offsets.
    for step in range(32):
        offset = (2 * math.pi * step) / 32

        orders.append((f"giant_angle_{step:02d}_fwd", angle_order(instance, offset, False)))
        orders.append((f"giant_angle_{step:02d}_rev", angle_order(instance, offset, True)))

    # Nearest-neighbor giant tours from selected starts.
    customers = [node for node in instance.coordinates.keys() if node != 0]

    depot_sorted = sorted(
        customers,
        key=lambda customer: dist[0][customer],
    )

    selected_starts = []

    selected_starts.extend(depot_sorted[:5])
    selected_starts.extend(depot_sorted[-5:])

    # Add some spread-out starts.
    step = max(1, len(depot_sorted) // 10)
    selected_starts.extend(depot_sorted[::step])

    seen_starts = []
    for customer in selected_starts:
        if customer not in seen_starts:
            seen_starts.append(customer)

    for customer in seen_starts:
        order = nearest_neighbor_customer_order(instance, customer, dist)
        orders.append((f"giant_nn_start_{customer}", order))

    candidates = []
    seen_signatures = set()

    for name, order in orders:
        solution = split_order_exact_k(order, instance, dist)

        if solution is None:
            continue

        signature = tuple(tuple(route) for route in solution)

        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)

        cost = total_cost(solution, dist)
        candidates.append((name, solution, cost))

    candidates.sort(key=lambda item: item[2])

    return candidates